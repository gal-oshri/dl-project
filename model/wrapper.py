"""
ConditionalLoRAModelWrapper: propagates input_ids to all ConditionalLoRA layers
so they can determine which positions correspond to the compression token.

Also handles saving/loading of the trained LoRA weights and SeparatedEmbedding.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn

from .conditional_lora import ConditionalLoRALinear
from .embedding import SeparatedEmbedding


class ConditionalLoRAModelWrapper(nn.Module):
    """
    Wrapper that propagates input_ids to all ConditionalLoRA layers,
    enabling position-aware conditional LoRA activation.
    """

    def __init__(self, model: nn.Module, compression_token_id: int):
        super().__init__()
        self.model = model
        self.compression_token_id = compression_token_id

    def _propagate_input_ids(self, input_ids: torch.Tensor) -> None:
        for module in self.model.modules():
            if isinstance(module, ConditionalLoRALinear):
                module._batch_input_ids = input_ids

    def _clear_input_ids(self) -> None:
        for module in self.model.modules():
            if isinstance(module, ConditionalLoRALinear):
                module._batch_input_ids = None

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        self._propagate_input_ids(input_ids)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        return outputs

    def generate(self, input_ids=None, **kwargs):
        if input_ids is not None:
            self._propagate_input_ids(input_ids)
        return self.model.generate(input_ids=input_ids, **kwargs)

    def save_pretrained(self, path: str) -> None:
        """Save LoRA weights and SeparatedEmbedding to disk."""
        os.makedirs(path, exist_ok=True)

        lora_state_dict = {}
        for name, module in self.model.named_modules():
            if isinstance(module, ConditionalLoRALinear):
                lora_state_dict[f"{name}.lora_A"] = module.lora_A.data.clone().cpu()
                lora_state_dict[f"{name}.lora_B"] = module.lora_B.data.clone().cpu()

        torch.save(lora_state_dict, os.path.join(path, "lora_weights.pt"))

        for name, module in self.model.named_modules():
            if isinstance(module, SeparatedEmbedding):
                torch.save(
                    module.new_embedding.state_dict(),
                    os.path.join(path, "sum_embedding.pt"),
                )
                break

    def load_pretrained(self, path: str) -> None:
        """Load LoRA weights and SeparatedEmbedding from disk."""
        device = next(self.model.parameters()).device

        lora_weights_path = os.path.join(path, "lora_weights.pt")
        lora_state_dict = torch.load(lora_weights_path, map_location=device, weights_only=True)

        lora_modules = {}
        for name, module in self.model.named_modules():
            if isinstance(module, ConditionalLoRALinear):
                lora_modules[name] = module

        loaded_count = 0
        for saved_key, saved_tensor in lora_state_dict.items():
            parts = saved_key.rsplit(".", 1)
            if len(parts) != 2:
                continue

            module_path, param_name = parts

            if module_path.startswith("model.") and module_path[6:] in lora_modules:
                module_path = module_path[6:]
            elif not module_path.startswith("model.") and f"model.{module_path}" in lora_modules:
                module_path = f"model.{module_path}"

            if module_path in lora_modules:
                module = lora_modules[module_path]
                target_dtype = module.lora_A.dtype
                if param_name == "lora_A":
                    module.lora_A = nn.Parameter(
                        saved_tensor.to(device=device, dtype=target_dtype)
                    )
                    loaded_count += 1
                elif param_name == "lora_B":
                    module.lora_B = nn.Parameter(
                        saved_tensor.to(device=device, dtype=target_dtype)
                    )
                    loaded_count += 1

        sum_embed_path = os.path.join(path, "sum_embedding.pt")
        if os.path.exists(sum_embed_path):
            sum_state_dict = torch.load(sum_embed_path, map_location=device, weights_only=True)
            for module in self.model.modules():
                if isinstance(module, SeparatedEmbedding):
                    module.new_embedding.load_state_dict(sum_state_dict)
                    break

        return loaded_count

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def config(self):
        return self.model.config

    def train(self, mode=True):
        self.model.train(mode)
        return super().train(mode)

    def eval(self):
        self.model.eval()
        return super().eval()
