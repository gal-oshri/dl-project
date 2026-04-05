"""
Model builder: assembles the full CCM model from components.

Pipeline:
1. Load base pretrained model and tokenizer
2. Add compression token to tokenizer
3. Wrap embedding with SeparatedEmbedding
4. Inject ConditionalLoRA into target attention layers
5. Wrap the model for input_ids propagation
"""

import logging
import sys
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training

from ..config import CCMConfig
from .embedding import SeparatedEmbedding
from .conditional_lora import ConditionalLoRALinear, inject_conditional_lora
from .wrapper import ConditionalLoRAModelWrapper

logger = logging.getLogger(__name__)


def ensure_hf_auth(model_name: str) -> None:
    """
    Ensure the user is authenticated with Hugging Face Hub.

    Many models (e.g. Llama-2) are gated and require:
    1. Accepting the license at the model page on huggingface.co
    2. Being logged in with a valid token

    This function checks the current login state and attempts interactive
    login if needed.
    """
    from huggingface_hub import HfApi, login
    from huggingface_hub.utils import HfHubHTTPError

    print(f"\n{'=' * 70}")
    print(f"  Hugging Face Authentication")
    print(f"{'=' * 70}")
    print(f"  Model requested: {model_name}")

    api = HfApi()
    user_info = None
    try:
        user_info = api.whoami()
        username = user_info.get("name", user_info.get("fullname", "unknown"))
        print(f"  Already logged in as: {username}")
    except Exception:
        print(f"  Not currently logged in.")
        print(f"\n  Attempting login...")
        print(f"  If prompted, paste your Hugging Face token.")
        print(f"  Get a token at: https://huggingface.co/settings/tokens")
        print(f"  (select 'Read' access is sufficient)\n")
        try:
            login()
            user_info = api.whoami()
            username = user_info.get("name", user_info.get("fullname", "unknown"))
            print(f"  Successfully logged in as: {username}")
        except Exception as e:
            print(f"\n  [ERROR] Hugging Face login failed: {e}")
            print(f"\n  To fix this, run one of the following BEFORE launching training:")
            print(f"    Option 1:  huggingface-cli login")
            print(f"    Option 2:  export HF_TOKEN=hf_your_token_here")
            print(f"    Option 3:  python -c \"from huggingface_hub import login; login()\"")
            print(f"\n  Get your token at: https://huggingface.co/settings/tokens")
            sys.exit(1)

    # Check if the model is accessible
    print(f"\n  Verifying access to model: {model_name}")
    try:
        api.model_info(model_name)
        print(f"  Model access confirmed.")
    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            print(f"\n  [ERROR] Access denied to model: {model_name}")
            print(f"\n  This is a GATED model. You must:")
            print(f"    1. Go to: https://huggingface.co/{model_name}")
            print(f"    2. Click 'Access repository' / accept the license agreement")
            print(f"    3. Wait for approval (usually instant for Llama models)")
            print(f"    4. Re-run this script")
            sys.exit(1)
        else:
            logger.warning("Could not verify model access (non-auth error): %s", e)
    except Exception as e:
        logger.warning("Could not verify model access: %s", e)

    print(f"{'=' * 70}\n")


def _detect_attention_modules(model: nn.Module) -> list:
    """Auto-detect attention projection layer names from the model architecture."""
    layer_endings = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or (
            "Linear" in type(module).__name__ and hasattr(module, "weight")
        ):
            ending = name.split(".")[-1]
            layer_endings.add(ending)

    attention_targets = []
    for ending in layer_endings:
        if any(
            x in ending.lower()
            for x in ["proj", "qkv", "query", "key", "value", "dense", "attention"]
        ):
            attention_targets.append(ending)

    return attention_targets or list(layer_endings)


def build_tokenizer(config: CCMConfig) -> Tuple[AutoTokenizer, int]:
    """Load tokenizer and add the compression token."""
    ensure_hf_auth(config.model.model_name)

    print(f"Loading tokenizer for: {config.model.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        trust_remote_code=config.model.trust_remote_code,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.add_special_tokens(
        {"additional_special_tokens": [config.model.compression_token]}
    )
    compression_token_id = tokenizer.convert_tokens_to_ids(config.model.compression_token)

    logger.info(
        "Tokenizer loaded. Vocab size: %d, compression token ID: %d",
        len(tokenizer),
        compression_token_id,
    )
    return tokenizer, compression_token_id


def build_model(config: CCMConfig) -> Tuple[ConditionalLoRAModelWrapper, AutoTokenizer, int]:
    """
    Build the complete CCM model pipeline.

    Returns:
        Tuple of (wrapped model, tokenizer, compression_token_id).
    """
    tokenizer, compression_token_id = build_tokenizer(config)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(config.model.torch_dtype, torch.float16)

    print(f"Loading base model: {config.model.model_name} (dtype={config.model.torch_dtype})")
    print(f"  This may take a few minutes for large models...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            trust_remote_code=config.model.trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=config.model.device_map,
        )
    except OSError as e:
        if "gated" in str(e).lower() or "401" in str(e) or "access" in str(e).lower():
            print(f"\n  [ERROR] Failed to download model: {config.model.model_name}")
            print(f"  The model is gated. Please:")
            print(f"    1. Visit https://huggingface.co/{config.model.model_name}")
            print(f"    2. Accept the license / request access")
            print(f"    3. Make sure you are logged in: huggingface-cli login")
            print(f"    4. Re-run this script")
            sys.exit(1)
        raise
    print(f"  Model loaded successfully on device: {model.device}")

    print(f"Setting up SeparatedEmbedding for compression token (ID={compression_token_id})")
    original_embedding = model.get_input_embeddings()
    separated_embedding = SeparatedEmbedding(original_embedding, compression_token_id)

    device = next(model.parameters()).device
    separated_embedding.new_embedding = separated_embedding.new_embedding.to(
        device=device, dtype=torch_dtype
    )
    model.set_input_embeddings(separated_embedding)

    model = prepare_model_for_kbit_training(model)

    for module in model.modules():
        if isinstance(module, SeparatedEmbedding):
            module.new_embedding.weight.requires_grad_(True)

    detected_targets = _detect_attention_modules(model)
    target_modules = config.lora.target_modules or detected_targets

    print(f"Injecting Conditional LoRA (r={config.lora.r}, alpha={config.lora.alpha})")
    print(f"  Target modules: {target_modules}")
    logger.info("LoRA target modules: %s", target_modules)

    model, replaced_modules = inject_conditional_lora(
        model=model,
        target_modules=target_modules,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        compression_token_id=compression_token_id,
    )
    print(f"  Injected into {len(replaced_modules)} layers")
    logger.info("Injected Conditional LoRA into %d layers", len(replaced_modules))

    wrapped_model = ConditionalLoRAModelWrapper(model, compression_token_id)

    total_params = sum(p.numel() for p in wrapped_model.parameters())
    trainable_params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
    print(f"\nParameter summary:")
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.4f}%)")
    logger.info(
        "Parameters — total: %s, trainable: %s (%.4f%%)",
        f"{total_params:,}",
        f"{trainable_params:,}",
        100 * trainable_params / total_params,
    )

    return wrapped_model, tokenizer, compression_token_id


def load_trained_model(
    config: CCMConfig,
    weights_path: str,
) -> Tuple[ConditionalLoRAModelWrapper, AutoTokenizer, int]:
    """
    Build model and load previously trained LoRA weights + embedding.

    Args:
        config: Model/LoRA configuration.
        weights_path: Directory containing lora_weights.pt and sum_embedding.pt.

    Returns:
        Tuple of (model with loaded weights, tokenizer, compression_token_id).
    """
    model, tokenizer, compression_token_id = build_model(config)
    loaded_count = model.load_pretrained(weights_path)
    logger.info("Loaded %d LoRA parameters from %s", loaded_count, weights_path)
    return model, tokenizer, compression_token_id
