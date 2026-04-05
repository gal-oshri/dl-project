"""
SeparatedEmbedding: trainable embedding for the compression token while keeping
the base vocabulary embedding frozen.

Follows the CCM paper's approach where the <COMP> (here <SUM>) token has its own
trainable embedding, ensuring the base model's vocabulary is unmodified.
"""

import torch
import torch.nn as nn


class SeparatedEmbedding(nn.Module):
    """
    Wraps the original embedding layer and provides a separate trainable
    embedding for the compression token. The base embedding stays frozen.

    During forward pass:
    - All tokens except the compression token use the frozen base embedding.
    - The compression token uses a dedicated trainable embedding.
    """

    def __init__(self, base_embedding: nn.Embedding, new_token_id: int):
        super().__init__()
        self.base_embedding = base_embedding
        self.new_token_id = new_token_id
        embed_dim = base_embedding.weight.shape[1]
        self.new_embedding = nn.Embedding(1, embed_dim)
        nn.init.normal_(self.new_embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        safe_ids = input_ids.clone()
        new_token_mask = input_ids == self.new_token_id
        safe_ids[new_token_mask] = 0

        embeds = self.base_embedding(safe_ids)

        if new_token_mask.any():
            new_embeds = self.new_embedding.weight[0]
            embeds = embeds.clone()
            embeds[new_token_mask] = new_embeds.to(embeds.dtype)

        return embeds

    @property
    def weight(self):
        return self.base_embedding.weight

    @property
    def num_embeddings(self):
        return self.base_embedding.num_embeddings

    @property
    def embedding_dim(self):
        return self.base_embedding.embedding_dim
