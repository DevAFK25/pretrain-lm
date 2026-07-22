import torch
from torch import nn

from model.self_attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Process each token representation independently."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=embedding_dim,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    """One decoder-only Transformer block."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feed_forward_dim: int,
    ) -> None:
        super().__init__()

        self.attention_norm = nn.LayerNorm(embedding_dim)

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        )

        self.feed_forward_norm = nn.LayerNorm(embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=feed_forward_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # First sub-layer: multi-head attention.
        normalized_x = self.attention_norm(x)

        attention_output, attention_weights = self.attention(
            normalized_x
        )

        # Residual connection.
        x = x + attention_output

        # Second sub-layer: feed-forward network.
        normalized_x = self.feed_forward_norm(x)

        feed_forward_output = self.feed_forward(normalized_x)

        # Second residual connection.
        x = x + feed_forward_output

        return x, attention_weights


def main() -> None:
    torch.manual_seed(42)

    sequence_length = 8
    embedding_dim = 32
    num_heads = 4
    feed_forward_dim = 128

    # Fake embedded sequence.
    x = torch.randn(
        sequence_length,
        embedding_dim,
    )

    block = TransformerBlock(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
    )

    output, attention_weights = block(x)

    print(f"Input shape:              {x.shape}")
    print(f"Output shape:             {output.shape}")
    print(f"Attention weights shape:  {attention_weights.shape}")


if __name__ == "__main__":
    main()