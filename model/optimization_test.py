import torch
from torch import nn

from language_model import SmallLanguageModel


def main() -> None:
    torch.manual_seed(42)

    vocab_size = 300
    block_size = 8
    embedding_dim = 32
    num_heads = 4
    feed_forward_dim = 128
    num_layers = 3
    learning_rate = 3e-4

    model = SmallLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        num_layers=num_layers,
    )

    # Fake sequence containing 9 tokens.
    tokens = torch.tensor(
        [10, 25, 41, 17, 98, 63, 14, 77, 52],
        dtype=torch.long,
    )

    # Input contains the first 8 tokens.
    input_ids = tokens[:-1]

    # Targets contain the next token for every input position.
    target_ids = tokens[1:]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    loss_function = nn.CrossEntropyLoss()

    print("Input IDs: ")
    print(input_ids)

    print("\nTarget IDs:")
    print(target_ids)

    # -------------------------
    # 1. Forward pass
    # -------------------------

    logits, _ = model(input_ids)

    # logits shape:  [sequence_length, vocab_size]
    # targets shape: [sequence_length]
    loss = loss_function(logits, target_ids)

    print(f"\nLoss before update: {loss.item():.4f}")

    # -------------------------
    # 2. Clear old gradients
    # -------------------------

    optimizer.zero_grad()

    # -------------------------
    # 3. Backpropagation
    # -------------------------

    loss.backward()

    # -------------------------
    # 4. Update parameters
    # -------------------------

    optimizer.step()

    # -------------------------
    # 5. Check the new loss
    # -------------------------

    with torch.no_grad():
        new_logits, _ = model(input_ids)
        new_loss = loss_function(new_logits, target_ids)

    print(f"Loss after update:  {new_loss.item():.4f}")


if __name__ == "__main__":
    main()