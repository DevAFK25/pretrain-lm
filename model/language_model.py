import torch
from torch import nn

from model.embeddings import GPTEmbeddings
from model.transformer import Transformer


class SmallLanguageModel(nn.Module):
    """A small decoder-only language model."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embedding_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.embeddings = GPTEmbeddings(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            block_size=block_size,
        )

        self.transformer = Transformer(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            feed_forward_dim=feed_forward_dim,
            num_layers=num_layers,
        )

        # Converts each final token representation into one score
        # for every token in the vocabulary.
        self.lm_head = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size,
            bias=False,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.embeddings(token_ids)

        x, attention_weights = self.transformer(x)

        logits = self.lm_head(x)

        return logits, attention_weights


def main() -> None:
    torch.manual_seed(42)

    vocab_size = 300
    block_size = 8
    embedding_dim = 32
    num_heads = 4
    feed_forward_dim = 128
    num_layers = 3

    # Eight fake token IDs.
    token_ids = torch.tensor(
        [10, 25, 41, 17, 98, 63, 14, 77],
        dtype=torch.long,
    )

    model = SmallLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        num_layers=num_layers,
    )

    logits, attention_weights = model(token_ids)

    print(f"Token IDs shape:          {token_ids.shape}")
    print(f"Logits shape:             {logits.shape}")
    print(f"Number of blocks:         {len(attention_weights)}")
    print(f"Attention shape/block:    {attention_weights[0].shape}")

    print("\nLogits for the first position:")
    print(logits[0])

    probabilities = torch.softmax(logits, dim=-1)

    print("\nProbability sum for each position:")
    print(probabilities.sum(dim=-1))

    predicted_token_ids = torch.argmax(
        probabilities,
        dim=-1,
    )

    print("\nPredicted next-token IDs:")
    print(predicted_token_ids)


if __name__ == "__main__":
    main()