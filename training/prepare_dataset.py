from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEXT_FILE = PROJECT_ROOT / "data" / "debug" / "sample.txt"
TOKENIZER_FILE = PROJECT_ROOT / "tokenizer" / "debug_tokenizer.json"

# Number of tokens the model can see at once.
BLOCK_SIZE = 8


class NextTokenDataset(Dataset):
    """Create fixed-length input-target pairs for next-token prediction."""

    def __init__(self, token_ids: list[int], block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be greater than zero.")

        if len(token_ids) < block_size + 1:
            raise ValueError(
                f"Need at least {block_size + 1} tokens, "
                f"but received only {len(token_ids)}."
            )

        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        # Each example needs block_size input tokens plus one next token.
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Read block_size + 1 tokens.
        chunk = self.token_ids[index : index + self.block_size + 1]

        # Input contains every token except the final one.
        inputs = chunk[:-1]

        # Targets contain every token except the first one.
        targets = chunk[1:]

        return inputs, targets


def load_token_ids() -> tuple[Tokenizer, list[int]]:
    """Load the tokenizer and encode the debug corpus."""

    if not TEXT_FILE.exists():
        raise FileNotFoundError(f"Text file not found: {TEXT_FILE}")

    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_FILE}\n"
            "Run tokenizer/train_tokenizer.py first."
        )

    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    text = TEXT_FILE.read_text(encoding="utf-8")

    encoding = tokenizer.encode(text)

    return tokenizer, encoding.ids


def inspect_example(
    tokenizer: Tokenizer,
    dataset: NextTokenDataset,
    index: int,
) -> None:
    """Print one dataset example in both token and text form."""

    inputs, targets = dataset[index]

    input_ids = inputs.tolist()
    target_ids = targets.tolist()

    input_tokens = [tokenizer.id_to_token(token_id) for token_id in input_ids]
    target_tokens = [tokenizer.id_to_token(token_id) for token_id in target_ids]

    print("=" * 70)
    print(f"Example index: {index}")
    print(f"Input IDs:     {input_ids}")
    print(f"Target IDs:    {target_ids}")
    print()
    print(f"Input tokens:  {input_tokens}")
    print(f"Target tokens: {target_tokens}")
    print()
    print(f"Input text:    {tokenizer.decode(input_ids)}")
    print(f"Target text:   {tokenizer.decode(target_ids)}")


def main() -> None:
    tokenizer, token_ids = load_token_ids()

    output_file = PROJECT_ROOT / "data" / "token_ids.pt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        torch.tensor(token_ids, dtype=torch.long),
        output_file,
    )

    print(f"Saved token IDs to: {output_file}")

    dataset = NextTokenDataset(
        token_ids=token_ids,
        block_size=BLOCK_SIZE,
    )

    print(f"Total corpus tokens: {len(token_ids)}")
    print(f"Block size:          {BLOCK_SIZE}")
    print(f"Dataset examples:    {len(dataset)}")
    print()

    inspect_example(
        tokenizer=tokenizer,
        dataset=dataset,
        index=0,
    )


if __name__ == "__main__":
    main()