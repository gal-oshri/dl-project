"""
Conditional LoRA: Low-Rank Adaptation that activates only at compression token positions.

From the CCM paper (Section 3.1, Figure 4):
    x'_h = W * x_h + m * ΔW * x_h
    where m = 1(x = <COMP>)

This ensures trainable parameters solely influence the model's compression
capabilities, preventing overfitting on inputs without considering the
compressed context memory.
"""

import torch
import torch.nn as nn
from typing import List


class ConditionalLoRALinear(nn.Module):
    """
    Linear layer with Conditional LoRA that only applies the low-rank
    adaptation when processing the compression token.

    For regular tokens:  output = W @ x          (frozen base weights)
    For <SUM> tokens:    output = (W + B @ A) @ x (base + LoRA adaptation)
    """

    def __init__(
        self,
        base_layer: nn.Module,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        compression_token_id: int = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.compression_token_id = compression_token_id
        self.scaling = lora_alpha / r

        for param in self.base_layer.parameters():
            param.requires_grad = False

        d_in = base_layer.in_features
        d_out = base_layer.out_features

        target_device = device or torch.device("cpu")
        target_dtype = dtype or torch.float32

        lora_a = torch.zeros(r, d_in, device=target_device, dtype=target_dtype)
        lora_b = torch.zeros(d_out, r, device=target_device, dtype=target_dtype)
        nn.init.kaiming_uniform_(lora_a, a=5**0.5)

        self.lora_A = nn.Parameter(lora_a)
        self.lora_B = nn.Parameter(lora_b)

        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        self._batch_input_ids = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)

        ids = self._batch_input_ids
        if ids is None or self.compression_token_id is None:
            return result

        if x.dim() != 3:
            return result

        batch_size, seq_len, _ = x.shape
        if ids.shape[0] != batch_size or ids.shape[1] != seq_len:
            return result

        comp_mask = (ids == self.compression_token_id).float().to(x.device)

        if comp_mask.sum() > 0:
            lora_out = self.dropout(x @ self.lora_A.T @ self.lora_B.T) * self.scaling
            mask_expanded = comp_mask.unsqueeze(-1)
            result = result + lora_out * mask_expanded

        return result


def inject_conditional_lora(
    model: nn.Module,
    target_modules: List[str],
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    compression_token_id: int,
) -> tuple:
    """
    Replace target Linear layers with ConditionalLoRALinear modules.

    Args:
        model: The base language model.
        target_modules: List of module name suffixes to replace (e.g., ["q_proj", "v_proj"]).
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout rate for LoRA.
        compression_token_id: Token ID of the compression token.

    Returns:
        Tuple of (modified model, list of replaced module paths).
    """
    replaced = []

    for name, module in list(model.named_modules()):
        module_name = name.split(".")[-1]
        if module_name not in target_modules:
            continue

        is_linear = isinstance(module, nn.Linear)
        module_class_name = type(module).__name__
        is_quantized_linear = "Linear" in module_class_name and hasattr(module, "weight")

        if not (is_linear or is_quantized_linear):
            continue

        parent_name = ".".join(name.split(".")[:-1])
        parent = model.get_submodule(parent_name) if parent_name else model

        try:
            if hasattr(module, "weight") and module.weight is not None:
                device = module.weight.device
                dtype = module.weight.dtype
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                dtype = torch.float16
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16

        conditional_lora = ConditionalLoRALinear(
            base_layer=module,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            compression_token_id=compression_token_id,
            device=device,
            dtype=dtype,
        )

        setattr(parent, module_name, conditional_lora)
        replaced.append(name)

    return model, replaced
