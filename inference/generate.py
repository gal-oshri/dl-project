"""
Inference with KV cache extraction for compressed context summarization.

Two-stage generation:
  Stage 1: Pass dialogue + <SUM> through model → extract <SUM>'s KV cache
  Stage 2: Use ONLY <SUM>'s KV cache (dialogue discarded) to generate summary

This demonstrates the core CCM compression: dialogue → <SUM> KV → summary,
where the entire dialogue is compressed into a single KV vector per layer.
"""

import logging

import torch
import torch.nn.functional as F
from transformers import DynamicCache, PreTrainedTokenizer

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


def generate_summary(
    model: ConditionalLoRAModelWrapper,
    tokenizer: PreTrainedTokenizer,
    dialogue: str,
    compression_token: str,
    compression_token_id: int,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    do_sample: bool = True,
    top_p: float = 0.9,
) -> str:
    """
    Generate a summary using the compressed context (KV cache extraction).

    Args:
        model: The CCM-wrapped model.
        tokenizer: Tokenizer with the compression token.
        dialogue: Input dialogue text.
        compression_token: The compression token string (e.g., "<SUM>").
        compression_token_id: Token ID of the compression token.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        do_sample: Whether to sample or use greedy decoding.
        top_p: Nucleus sampling probability.

    Returns:
        Generated summary string.
    """
    model.eval()
    base_model = model.model
    device = model.device

    # Stage 1: encode dialogue + <SUM>
    prompt = f"{dialogue}\n{compression_token}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"][0]
    comp_pos = (input_ids == compression_token_id).nonzero(as_tuple=True)[0]
    if len(comp_pos) == 0:
        logger.warning("Compression token not found in input — returning empty summary")
        return ""
    comp_pos = comp_pos[0].item()

    with torch.no_grad():
        model._propagate_input_ids(inputs["input_ids"])
        outputs = base_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=True,
            return_dict=True,
        )
        full_kv_cache = outputs.past_key_values

    # Stage 2: extract <SUM> KV and generate autoregressively
    num_layers = len(full_kv_cache)
    comp_kv_cache = DynamicCache()

    for layer_idx in range(num_layers):
        key, value = _get_kv_layer(full_kv_cache, layer_idx)
        comp_key = key[:, :, comp_pos : comp_pos + 1, :].clone()
        comp_value = value[:, :, comp_pos : comp_pos + 1, :].clone()
        comp_kv_cache.update(comp_key, comp_value, layer_idx)

    generated_ids = []
    current_kv = comp_kv_cache
    next_token_input = tokenizer.encode(
        "\n", add_special_tokens=False, return_tensors="pt"
    ).to(device)

    for _ in range(max_new_tokens):
        kv_len = current_kv.get_seq_length()
        input_len = next_token_input.shape[1]
        attn_mask = torch.ones(1, kv_len + input_len, device=device)

        model._propagate_input_ids(next_token_input)

        with torch.no_grad():
            outputs = base_model(
                input_ids=next_token_input,
                attention_mask=attn_mask,
                past_key_values=current_kv,
                use_cache=True,
                return_dict=True,
            )

        logits = outputs.logits[:, -1, :]

        if do_sample:
            logits = logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        if next_token.item() == tokenizer.eos_token_id:
            break

        generated_ids.append(next_token.item())
        next_token_input = next_token
        current_kv = outputs.past_key_values

    return tokenizer.decode(generated_ids, skip_special_tokens=True)
