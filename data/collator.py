"""
Data collator for CCM training.

Keeps input (dialogue + <SUM>) and target (summary) SEPARATE so the training
loop can:
1. Encode input_ids → get full KV cache
2. Extract ONLY the <SUM> token's KV cache
3. Use <SUM> KV as context to predict target_ids via teacher forcing
"""

import torch
from transformers import PreTrainedTokenizer


class CompressionDataCollator:
    """
    Collates dialogue-summary pairs into batches with separated input/target tensors.

    Input format:  dialogue tokens + newline + <SUM> token (padded to max_input_length)
    Target format: summary tokens (padded to max_target_length)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        compression_token_id: int,
        max_input_length: int = 400,
        max_target_length: int = 100,
    ):
        self.tokenizer = tokenizer
        self.compression_token_id = compression_token_id
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __call__(self, features: list) -> dict:
        batch_input_ids = []
        batch_input_attention = []
        batch_target_ids = []
        batch_target_attention = []

        for feature in features:
            dialogue = feature["dialogue"]
            summary = feature["summary"]

            dialogue_encoded = self.tokenizer(
                dialogue,
                truncation=True,
                max_length=self.max_input_length - 2,
                padding=False,
                add_special_tokens=False,
                return_tensors=None,
            )

            newline_id = self.tokenizer.encode("\n", add_special_tokens=False)
            input_ids = (
                dialogue_encoded["input_ids"]
                + newline_id
                + [self.compression_token_id]
            )

            actual_len = len(input_ids)
            pad_length = self.max_input_length - actual_len
            if pad_length > 0:
                input_attention = [1] * actual_len + [0] * pad_length
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
            else:
                input_ids = input_ids[: self.max_input_length]
                input_attention = [1] * self.max_input_length

            target_encoded = self.tokenizer(
                summary,
                truncation=True,
                max_length=self.max_target_length,
                padding="max_length",
                return_tensors=None,
            )

            batch_input_ids.append(input_ids)
            batch_input_attention.append(input_attention)
            batch_target_ids.append(target_encoded["input_ids"])
            batch_target_attention.append(target_encoded["attention_mask"])

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "input_attention_mask": torch.tensor(batch_input_attention, dtype=torch.long),
            "target_ids": torch.tensor(batch_target_ids, dtype=torch.long),
            "target_attention_mask": torch.tensor(batch_target_attention, dtype=torch.long),
        }
