import torch
from torch import nn

from model.transformer_block import TransformerBlock


class Transformer(nn.Module):
    """Stack multiple Transformer blocks."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    feed_forward_dim=feed_forward_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Modern decoder-only transformers finish with one final LayerNorm.
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        all_attention_weights = []

        for block in self.blocks:
            x, attention_weights = block(x)
            all_attention_weights.append(attention_weights)

        x = self.final_layer_norm(x)

        return x, all_attention_weights


def main() -> None:
    torch.manual_seed(42)

    sequence_length = 8
    embedding_dim = 32
    num_heads = 4
    feed_forward_dim = 128
    num_layers = 3

    x = torch.randn(
        sequence_length,
        embedding_dim,
    )

    transformer = Transformer(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        num_layers=num_layers,
    )

    output, attention_weights = transformer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of blocks: {len(attention_weights)}")

    print("\nAttention shape from Block 1:")
    print(attention_weights[0].shape)

    print("\nAttention shape from Block 2:")
    print(attention_weights[1].shape)

    print("\nAttention shape from Block 3:")
    print(attention_weights[2].shape)


if __name__ == "__main__":
    main()