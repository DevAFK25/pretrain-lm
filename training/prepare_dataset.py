from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

WIKITEXT_TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "wikitext_train.pt"
)

class NextTokenDataset(Dataset):
    """
    Creates non-overlapping sequences for next-token prediction.

    Example with block_size=4:

    input:  [0, 1, 2, 3]   target: [1, 2, 3, 4]
    input:  [4, 5, 6, 7]   target: [5, 6, 7, 8]
    """

    def __init__(
        self,
        token_ids: torch.Tensor,
        block_size: int,
    ) -> None:
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("token_ids must be a torch.Tensor.")

        if token_ids.ndim != 1:
            raise ValueError(
                "token_ids must be one-dimensional. "
                f"Received shape: {tuple(token_ids.shape)}"
            )

        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "token_ids must use torch.int32 or torch.int64. "
                f"Received dtype: {token_ids.dtype}"
            )

        if block_size <= 0:
            raise ValueError("block_size must be greater than zero.")

        if token_ids.numel() <= block_size:
            raise ValueError(
                "The corpus must contain more tokens than block_size."
            )

        self.token_ids = token_ids
        self.block_size = block_size

        # Each example requires block_size inputs plus one target token.
        self.num_examples = (
            token_ids.numel() - 1
        ) // block_size

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.num_examples:
            raise IndexError(
                f"Dataset index {index} is out of range."
            )

        start = index * self.block_size
        end = start + self.block_size

        input_ids = self.token_ids[start:end]
        target_ids = self.token_ids[start + 1:end + 1]

        # Embedding layers and cross-entropy targets require int64.
        return input_ids.long(), target_ids.long()


def main() -> None:
    block_size = 8

    if not WIKITEXT_TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Tokenized WikiText file not found:\n"
            f"{WIKITEXT_TRAIN_FILE}"
        )

    token_ids = torch.load(
        WIKITEXT_TRAIN_FILE,
        map_location="cpu",
        weights_only=True,
    )

    dataset = NextTokenDataset(
        token_ids=token_ids,
        block_size=block_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
    )

    input_batch, target_batch = next(iter(dataloader))

    input_ids, target_ids = dataset[0]

    shift_is_correct = torch.equal(
        input_ids[1:],
        target_ids[:-1],
    )

    print(f"Input batch shape:    {tuple(input_batch.shape)}")
    print(f"Target batch shape:   {tuple(target_batch.shape)}")
    print(f"Input batch dtype:    {input_batch.dtype}")
    print(f"Target batch dtype:   {target_batch.dtype}")

    if not shift_is_correct:
        raise RuntimeError("Input-target shift validation failed.")

    print("\nDataset validation passed.")


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    token_file = (
        project_root
        / "data"
        / "tokenized"
        / "wikitext_train.pt"
    )

    token_ids = torch.load(
        token_file,
        map_location="cpu",
        weights_only=True,
    )

    block_size = 128
    dataset = NextTokenDataset(
        token_ids=token_ids,
        block_size=block_size,
    )

    first_input, first_target = dataset[0]
    second_input, second_target = dataset[1]

    print(f"Corpus tokens:       {token_ids.numel():,}")
    print(f"Dataset examples:    {len(dataset):,}")
    print(f"Input shape:         {tuple(first_input.shape)}")
    print(f"Target shape:        {tuple(first_target.shape)}")
    print(f"Shift correct:       {torch.equal(first_input[1:], first_target[:-1])}")
    print(
        "Blocks separated:   "
        f"{torch.equal(second_input, token_ids[block_size:2 * block_size].long())}"
    )

    assert first_input.shape == (block_size,)
    assert first_target.shape == (block_size,)
    assert torch.equal(first_input[1:], first_target[:-1])
    assert torch.equal(
        second_input,
        token_ids[block_size:2 * block_size].long(),
    )

    print("Dataset validation passed.")