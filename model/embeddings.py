import torch
from torch import nn


class GPTEmbeddings(nn.Module):
    """
    Combines token embeddings and positional embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        block_size: int,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
        )

        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.shape[-1]

        if sequence_length > self.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"block size {self.block_size}."
            )

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(positions)

        return token_embeddings + position_embeddings