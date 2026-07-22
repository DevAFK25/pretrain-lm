from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.prepare_dataset import NextTokenDataset
from model.language_model import SmallLanguageModel
from tokenizers import Tokenizer


def generate_text(
    model: SmallLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    block_size: int,
    device: torch.device,
    max_new_tokens: int = 30,
) -> str:
    """Generate text one token at a time."""

    model.eval()

    encoding = tokenizer.encode(prompt)
    generated_ids = encoding.ids

    if not generated_ids:
        raise ValueError("The prompt produced no token IDs.")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # The model can only see block_size tokens at once.
            context_ids = generated_ids[-block_size:]

            # Shape before unsqueeze:
            # [sequence_length]
            #
            # Shape after unsqueeze:
            # [1, sequence_length]
            input_ids = torch.tensor(
                context_ids,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            logits, _ = model(input_ids)

            # Take the logits from the final position.
            next_token_logits = logits[0, -1, :]

            # Select the token with the highest score.
            next_token_id = torch.argmax(
                next_token_logits,
            ).item()

            generated_ids.append(next_token_id)

    return tokenizer.decode(generated_ids)


def main() -> None:
    torch.manual_seed(42)

    # -------------------------
    # Configuration
    # -------------------------

    vocab_size = 300
    block_size = 8
    batch_size = 4

    embedding_dim = 32
    num_heads = 4
    feed_forward_dim = 128
    num_layers = 3

    # -------------------------
    # Device
    # -------------------------

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # -------------------------
    # Load token IDs
    # -------------------------

    project_root = Path(__file__).resolve().parent
    token_file = project_root / "data" / "token_ids.pt"

    if not token_file.exists():
        raise FileNotFoundError(
            f"Could not find token file: {token_file}\n"
            "Run training/prepare_dataset.py first."
        )

    token_ids = torch.load(
        token_file,
        map_location="cpu",
        weights_only=True,
    )

    token_ids = torch.as_tensor(
        token_ids,
        dtype=torch.long,
    )

    print(f"Number of corpus tokens: {len(token_ids)}")

    # -------------------------
    # Dataset and DataLoader
    # -------------------------

    dataset = NextTokenDataset(
        token_ids=token_ids.tolist(),
        block_size=block_size,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    print(f"Number of dataset examples: {len(dataset)}")
    print(f"Number of batches: {len(data_loader)}")

    # -------------------------
    # Model
    # -------------------------

    model = SmallLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        num_layers=num_layers,
    ).to(device)

    # -------------------------
    # Training configuration
    # -------------------------

    learning_rate = 3e-4
    num_epochs = 5

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    loss_function = torch.nn.CrossEntropyLoss()

    # -------------------------
    # Training loop
    # -------------------------

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for step, (input_ids, target_ids) in enumerate(data_loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Remove gradients left over from the previous step.
            optimizer.zero_grad()

            # Forward pass.
            logits, attention = model(input_ids)

            # logits shape:
            # [batch_size, sequence_length, vocab_size]
            #
            # CrossEntropyLoss expects:
            # [number_of_predictions, vocab_size]

            loss = loss_function(
                logits.reshape(-1, vocab_size),
                target_ids.reshape(-1),
            )

            # Calculate gradients.
            loss.backward()

            # Update the model's weights.
            optimizer.step()

            total_loss += loss.item()

            if step % 20 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Step {step}/{len(data_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        average_loss = total_loss / len(data_loader)

        print(
            f"\nEpoch {epoch + 1} complete | "
            f"Average loss: {average_loss:.4f}\n"
        )
    
    # -------------------------
    # Generate text
    # -------------------------

    tokenizer_file = (
        project_root
        / "tokenizer"
        / "debug_tokenizer.json"
    )

    tokenizer = Tokenizer.from_file(
        str(tokenizer_file)
    )

    prompt = "The"

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        block_size=block_size,
        device=device,
        max_new_tokens=30,
    )

    print("=" * 70)
    print("Prompt:")
    print(prompt)
    print()
    print("Generated text:")
    print(generated_text)


if __name__ == "__main__":
    main()