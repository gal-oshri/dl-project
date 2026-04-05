"""
Custom training loop for CCM compression training.

Training flow per step (following CCM paper Algorithm 1):
1. Encode dialogue + <SUM> → get full KV cache
2. Extract ONLY the <SUM> token's KV cache (discard dialogue KV)
3. Use <SUM> KV as sole context to predict summary via teacher forcing
4. Compute cross-entropy loss on summary tokens
5. Backpropagate through <SUM> KV → update only LoRA weights
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import DynamicCache, PreTrainedTokenizer
from tqdm import tqdm

from ..model.wrapper import ConditionalLoRAModelWrapper

logger = logging.getLogger(__name__)


def _get_kv_layer(cache, layer_idx):
    """Extract (key, value) tensors from a KV cache layer."""
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    try:
        item = cache[layer_idx]
        if isinstance(item, (tuple, list)):
            return item[0], item[1]
    except (TypeError, IndexError):
        pass
    items = list(cache)
    k, v, *_ = items[layer_idx]
    return k, v


def _extract_compression_kv(
    full_kv_cache,
    input_ids: torch.Tensor,
    compression_token_id: int,
    pad_token_id: int,
    device: torch.device,
) -> Optional[DynamicCache]:
    """
    Find <SUM> positions in each sample and extract their KV vectors
    into a DynamicCache.

    Returns None if <SUM> is missing from any sample.
    """
    batch_size = input_ids.shape[0]
    comp_positions = []

    for b in range(batch_size):
        sample_ids = input_ids[b]
        non_pad_mask = sample_ids != pad_token_id
        comp_mask = sample_ids == compression_token_id
        positions = (comp_mask & non_pad_mask).nonzero(as_tuple=True)[0]

        if len(positions) == 0:
            return None
        comp_positions.append(positions[-1].item())

    num_layers = len(full_kv_cache)
    comp_kv_cache = DynamicCache()

    for layer_idx in range(num_layers):
        key, value = _get_kv_layer(full_kv_cache, layer_idx)
        comp_keys = []
        comp_values = []

        for b in range(batch_size):
            pos = comp_positions[b]
            comp_keys.append(key[b : b + 1, :, pos : pos + 1, :].clone())
            comp_values.append(value[b : b + 1, :, pos : pos + 1, :].clone())

        stacked_key = torch.cat(comp_keys, dim=0).to(device)
        stacked_value = torch.cat(comp_values, dim=0).to(device)
        comp_kv_cache.update(stacked_key, stacked_value, layer_idx)

    return comp_kv_cache


def train_step(
    model: ConditionalLoRAModelWrapper,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    compression_token_id: int,
    tokenizer: PreTrainedTokenizer,
) -> Optional[float]:
    """
    Execute one training step.

    Returns:
        Loss value as a float, or None if the batch was skipped.
    """
    model.train()
    device = model.device
    base_model = model.model

    input_ids = batch["input_ids"].to(device)
    input_attention = batch["input_attention_mask"].to(device)
    target_ids = batch["target_ids"].to(device)
    target_attention = batch["target_attention_mask"].to(device)
    batch_size = input_ids.shape[0]

    # Stage 1: encode dialogue + <SUM>
    model._propagate_input_ids(input_ids)
    encoder_outputs = base_model(
        input_ids=input_ids,
        attention_mask=input_attention,
        use_cache=True,
        return_dict=True,
    )

    # Stage 2: extract <SUM> KV cache
    comp_kv_cache = _extract_compression_kv(
        encoder_outputs.past_key_values,
        input_ids,
        compression_token_id,
        tokenizer.pad_token_id,
        device,
    )
    if comp_kv_cache is None:
        return None

    # Stage 3: decode summary using <SUM> KV as sole context
    kv_len = 1
    target_len = target_ids.shape[1]
    full_attention = torch.ones(batch_size, kv_len + target_len, device=device)

    for b in range(batch_size):
        pad_start = target_attention[b].sum().item()
        full_attention[b, kv_len + pad_start :] = 0

    model._propagate_input_ids(target_ids)
    outputs = base_model(
        input_ids=target_ids,
        attention_mask=full_attention,
        past_key_values=comp_kv_cache,
        use_cache=False,
        return_dict=True,
    )

    # Stage 4: compute loss
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = target_ids[:, 1:].contiguous()

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(
    model: ConditionalLoRAModelWrapper,
    dataloader: DataLoader,
    compression_token_id: int,
    tokenizer: PreTrainedTokenizer,
) -> float:
    """Evaluate model on a dataloader, returning average loss."""
    model.eval()
    device = model.device
    base_model = model.model
    losses = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        try:
            input_ids = batch["input_ids"].to(device)
            input_attention = batch["input_attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_attention = batch["target_attention_mask"].to(device)
            batch_size = input_ids.shape[0]

            model._propagate_input_ids(input_ids)
            encoder_outputs = base_model(
                input_ids=input_ids,
                attention_mask=input_attention,
                use_cache=True,
                return_dict=True,
            )

            comp_kv_cache = _extract_compression_kv(
                encoder_outputs.past_key_values,
                input_ids,
                compression_token_id,
                tokenizer.pad_token_id,
                device,
            )
            if comp_kv_cache is None:
                continue

            target_len = target_ids.shape[1]
            full_attention = torch.ones(batch_size, 1 + target_len, device=device)

            model._propagate_input_ids(target_ids)
            outputs = base_model(
                input_ids=target_ids,
                attention_mask=full_attention,
                past_key_values=comp_kv_cache,
                use_cache=False,
                return_dict=True,
            )

            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            losses.append(loss.item())

        except Exception:
            logger.warning("Skipping batch during evaluation", exc_info=True)
            continue

    return sum(losses) / len(losses) if losses else float("inf")


def run_training(
    model: ConditionalLoRAModelWrapper,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    compression_token_id: int,
    tokenizer: PreTrainedTokenizer,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    output_dir: str = "./compressed_context_model",
    log_interval: int = 50,
) -> dict:
    """
    Full training loop with validation and model saving.

    Returns:
        Dictionary with training history (train_losses, val_losses, best_loss).
    """
    lora_params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    logger.info(
        "Training — epochs: %d, batch count: %d, lr: %s, trainable params: %s",
        num_epochs,
        len(train_dataloader),
        learning_rate,
        f"{sum(p.numel() for p in lora_params):,}",
    )

    best_val_loss = float("inf")
    history = {"train_losses": [], "val_losses": [], "best_loss": float("inf")}

    for epoch in range(num_epochs):
        epoch_losses = []
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            try:
                loss = train_step(model, batch, optimizer, compression_token_id, tokenizer)

                if loss is None:
                    continue

                epoch_losses.append(loss)
                scheduler.step()

                avg_loss = sum(epoch_losses[-100:]) / min(len(epoch_losses), 100)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            except Exception:
                logger.error("Error in batch %d", batch_idx, exc_info=True)
                continue

        if not epoch_losses:
            logger.error("No successful batches in epoch %d, stopping.", epoch + 1)
            break

        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
        history["train_losses"].append(epoch_avg_loss)

        val_loss = evaluate(model, val_dataloader, compression_token_id, tokenizer)
        history["val_losses"].append(val_loss)

        logger.info(
            "Epoch %d — train loss: %.4f, val loss: %.4f",
            epoch + 1,
            epoch_avg_loss,
            val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_loss"] = best_val_loss
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info("New best model saved to %s (val loss: %.4f)", output_dir, val_loss)

    return history
