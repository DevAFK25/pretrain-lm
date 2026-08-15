import math

import torch
from torch import nn


EMBEDDING_DIM = 32
NUM_HEADS = 4


class SelfAttentionHead(nn.Module):
    """One causal self-attention head."""

    def __init__(
        self,
        embedding_dim: int,
        head_dim: int,
    ) -> None:
        super().__init__()

        self.head_dim = head_dim

        # Every head sees the complete embedding and independently
        # projects it into a smaller head-dimensional space.
        self.query = nn.Linear(
            in_features=embedding_dim,
            out_features=head_dim,
            bias=False,
        )

        self.key = nn.Linear(
            in_features=embedding_dim,
            out_features=head_dim,
            bias=False,
        )

        self.value = nn.Linear(
            in_features=embedding_dim,
            out_features=head_dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor with shape [sequence_length, embedding_dim]

        Returns:
            context: Tensor with shape [sequence_length, head_dim]
            attention_weights:
                Tensor with shape [sequence_length, sequence_length]
        """

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Shape:
        # [sequence_length, head_dim]
        # @
        # [head_dim, sequence_length]
        #
        # Result:
        # [sequence_length, sequence_length]
        scores = queries @ keys.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

        sequence_length = x.shape[-2]

        # True values represent forbidden future positions.
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        # Weighted combination of Value vectors.
        #
        # [sequence_length, sequence_length]
        # @
        # [sequence_length, head_dim]
        #
        # Result:
        # [sequence_length, head_dim]
        context = attention_weights @ values

        return context, attention_weights


class MultiHeadAttention(nn.Module):
    """Run several self-attention heads in parallel."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    embedding_dim=embedding_dim,
                    head_dim=self.head_dim,
                )
                for _ in range(num_heads)
            ]
        )

        # Mix information produced by all heads.
        self.output_projection = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor with shape [sequence_length, embedding_dim]

        Returns:
            output: Tensor with shape [sequence_length, embedding_dim]
            all_attention_weights:
                Tensor with shape
                [num_heads, sequence_length, sequence_length]
        """

        head_outputs = []
        head_attention_weights = []

        for head in self.heads:
            context, attention_weights = head(x)

            head_outputs.append(context)
            head_attention_weights.append(attention_weights)

        # Each context has shape:
        # [sequence_length, head_dim]
        #
        # Concatenation produces:
        # [sequence_length, num_heads * head_dim]
        #
        # Here:
        # [sequence_length, 4 * 8]
        # =
        # [sequence_length, 32]
        concatenated = torch.cat(
            head_outputs,
            dim=-1,
        )

        output = self.output_projection(concatenated)

        # Keep each head's attention matrix separate for inspection.
        #
        # Shape:
        # [num_heads, sequence_length, sequence_length]
        all_attention_weights = torch.stack(
            head_attention_weights,
            dim=0,
        )

        return output, all_attention_weights


def main() -> None:
    torch.manual_seed(42)

    sequence_length = 4

    # Pretend these are embeddings for:
    # "The patient feels better"
    #
    # Shape:
    # [4 tokens, 32 embedding dimensions]
    x = torch.randn(
        sequence_length,
        EMBEDDING_DIM,
    )

    attention = MultiHeadAttention(
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
    )

    output, attention_weights = attention(x)

    print(f"Input shape:              {x.shape}")
    print(f"Number of heads:          {NUM_HEADS}")
    print(f"Dimension per head:       {EMBEDDING_DIM // NUM_HEADS}")
    print(f"Output shape:             {output.shape}")
    print(f"Attention weights shape:  {attention_weights.shape}")

    print("\nAttention weights from Head 1:")
    print(attention_weights[0])

    print("\nAttention weights from Head 2:")
    print(attention_weights[1])

    print("\nRow sums for Head 1:")
    print(attention_weights[0].sum(dim=-1))

    print("\nFinal multi-head output:")
    print(output)


if __name__ == "__main__":
    main()