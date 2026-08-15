import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_FILE = (
    PROJECT_ROOT
    / "tokenizer"
    / "output"
    / "tokenizer.json"
)

WIKITEXT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wikitext-103"
)

ALLEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "homeopathy"
    / "allens_encyclopedia_i_x_clean.txt"
)

OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "tokenized"

TEMP_DIRECTORY = OUTPUT_DIRECTORY / ".temporary"

END_OF_TEXT_TOKEN = "<|endoftext|>"
END_OF_TEXT_BYTES = END_OF_TEXT_TOKEN.encode("utf-8")

# Amount of raw input read at once.
READ_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB

# Number of token IDs checked at once while hashing.
HASH_CHUNK_SIZE = 1_000_000


def load_tokenizer() -> Tokenizer:
    """Load and validate the final tokenizer."""

    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_FILE}"
        )

    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

    end_of_text_id = tokenizer.token_to_id(END_OF_TEXT_TOKEN)

    if end_of_text_id is None:
        raise ValueError(
            f"The tokenizer does not contain {END_OF_TEXT_TOKEN}."
        )

    encoded_marker = tokenizer.encode(
        END_OF_TEXT_TOKEN,
        add_special_tokens=False,
    ).ids

    if encoded_marker != [end_of_text_id]:
        raise ValueError(
            f"{END_OF_TEXT_TOKEN} must encode as exactly one token, "
            f"but produced: {encoded_marker}"
        )

    return tokenizer


def file_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file without loading it all at once."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(READ_CHUNK_SIZE):
            sha256.update(chunk)

    return sha256.hexdigest()


def yield_documents(input_file: Path):
    """
    Yield exact UTF-8 byte segments ending at <|endoftext|> boundaries.

    Concatenating every yielded segment recreates the original file exactly.
    """

    carry = b""

    with input_file.open("rb") as file:
        while raw_chunk := file.read(READ_CHUNK_SIZE):
            carry += raw_chunk

            while True:
                marker_index = carry.find(END_OF_TEXT_BYTES)

                if marker_index == -1:
                    break

                segment_end = marker_index + len(END_OF_TEXT_BYTES)

                segment = carry[:segment_end]
                carry = carry[segment_end:]

                yield segment

    # Preserve any text after the final <|endoftext|>.
    if carry:
        yield carry


def tensor_sha256(token_tensor: torch.Tensor) -> str:
    """Hash a token tensor in chunks without creating one huge Python list."""

    sha256 = hashlib.sha256()

    for start in range(0, token_tensor.numel(), HASH_CHUNK_SIZE):
        end = min(
            start + HASH_CHUNK_SIZE,
            token_tensor.numel(),
        )

        chunk = (
            token_tensor[start:end]
            .contiguous()
            .numpy()
            .astype("<i4", copy=False)
        )

        sha256.update(chunk.tobytes())

    return sha256.hexdigest()


def tokenize_file(
    tokenizer: Tokenizer,
    input_file: Path,
    output_file: Path,
) -> dict[str, int | str]:
    """
    Tokenize one corpus safely.

    Raw text is read incrementally, token IDs are written to a temporary
    binary file, and the final result is saved as one PyTorch tensor.
    """

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    if input_file.stat().st_size == 0:
        raise ValueError(
            f"Input file is empty: {input_file}"
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    TEMP_DIRECTORY.mkdir(parents=True, exist_ok=True)

    temporary_file = (
        TEMP_DIRECTORY
        / f"{output_file.stem}.tokens.bin"
    )

    if temporary_file.exists():
        temporary_file.unlink()

    source_hash = file_sha256(input_file)

    reconstructed_text_hash = hashlib.sha256()
    token_hash = hashlib.sha256()

    vocabulary_size = tokenizer.get_vocab_size()

    total_tokens = 0
    document_count = 0
    minimum_token_id: int | None = None
    maximum_token_id: int | None = None

    try:
        with temporary_file.open("wb") as token_output:
            for document_bytes in yield_documents(input_file):
                reconstructed_text_hash.update(document_bytes)

                try:
                    document_text = document_bytes.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise UnicodeError(
                        f"Invalid UTF-8 in {input_file}: {error}"
                    ) from error

                token_ids = tokenizer.encode(
                    document_text,
                    add_special_tokens=False,
                ).ids

                if not token_ids:
                    continue

                token_array = np.asarray(
                    token_ids,
                    dtype="<i4",
                )

                local_minimum = int(token_array.min())
                local_maximum = int(token_array.max())

                if local_minimum < 0:
                    raise ValueError(
                        f"Negative token ID found in {input_file}."
                    )

                if local_maximum >= vocabulary_size:
                    raise ValueError(
                        f"Token ID {local_maximum} exceeds vocabulary "
                        f"size {vocabulary_size}."
                    )

                token_bytes = token_array.tobytes()

                token_output.write(token_bytes)
                token_hash.update(token_bytes)

                total_tokens += len(token_array)
                document_count += 1

                if minimum_token_id is None:
                    minimum_token_id = local_minimum
                else:
                    minimum_token_id = min(
                        minimum_token_id,
                        local_minimum,
                    )

                if maximum_token_id is None:
                    maximum_token_id = local_maximum
                else:
                    maximum_token_id = max(
                        maximum_token_id,
                        local_maximum,
                    )

                if document_count % 1_000 == 0:
                    print(
                        f"  Documents processed: {document_count:,} | "
                        f"Tokens written: {total_tokens:,}"
                    )

        # This proves that streaming did not drop, duplicate, or reorder text.
        reconstructed_hash = reconstructed_text_hash.hexdigest()

        if reconstructed_hash != source_hash:
            raise RuntimeError(
                "Text-integrity check failed. The streamed segments do not "
                "reconstruct the original source file."
            )

        if total_tokens == 0:
            raise ValueError(
                f"No token IDs were produced for {input_file}."
            )

        expected_temporary_size = total_tokens * 4
        actual_temporary_size = temporary_file.stat().st_size

        if actual_temporary_size != expected_temporary_size:
            raise RuntimeError(
                "Temporary token file has an unexpected size.\n"
                f"Expected: {expected_temporary_size:,} bytes\n"
                f"Actual:   {actual_temporary_size:,} bytes"
            )

        # Memory-map the temporary token file rather than reading the whole
        # thing into a Python list.
        mapped_tensor = torch.from_file(
            str(temporary_file),
            shared=False,
            size=total_tokens,
            dtype=torch.int32,
        )

        torch.save(
            mapped_tensor,
            output_file,
        )

        # Reload and verify the final .pt file before deleting the temporary.
        saved_tensor = torch.load(
            output_file,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(saved_tensor, torch.Tensor):
            raise TypeError(
                f"{output_file} did not contain a PyTorch tensor."
            )

        if saved_tensor.dtype != torch.int32:
            raise TypeError(
                f"Expected torch.int32, received {saved_tensor.dtype}."
            )

        if saved_tensor.ndim != 1:
            raise ValueError(
                f"Expected a 1D tensor, received shape "
                f"{tuple(saved_tensor.shape)}."
            )

        if saved_tensor.numel() != total_tokens:
            raise RuntimeError(
                "Final tensor token count does not match the number written."
            )

        final_token_hash = tensor_sha256(saved_tensor)
        original_token_hash = token_hash.hexdigest()

        if final_token_hash != original_token_hash:
            raise RuntimeError(
                "Token-integrity check failed after saving the final tensor."
            )

        statistics = {
            "input_file": str(
                input_file.relative_to(PROJECT_ROOT)
            ),
            "output_file": str(
                output_file.relative_to(PROJECT_ROOT)
            ),
            "source_file_size_bytes": input_file.stat().st_size,
            "source_sha256": source_hash,
            "document_segments": document_count,
            "token_count": total_tokens,
            "minimum_token_id": minimum_token_id,
            "maximum_token_id": maximum_token_id,
            "token_sha256": final_token_hash,
            "tensor_dtype": str(saved_tensor.dtype),
        }

        # Temporary data is removed only after every check succeeds.
        temporary_file.unlink()

        return statistics

    except Exception:
        print(
            "\nTokenization failed. The temporary file was preserved at:"
        )
        print(temporary_file)
        raise


def inspect_saved_tokens(
    tokenizer: Tokenizer,
    token_file: Path,
    number_of_tokens: int = 80,
) -> None:
    """Decode the beginning and end of a saved token tensor."""

    token_tensor = torch.load(
        token_file,
        map_location="cpu",
        weights_only=True,
    )

    beginning_ids = (
        token_tensor[:number_of_tokens]
        .to(torch.long)
        .tolist()
    )

    ending_ids = (
        token_tensor[-number_of_tokens:]
        .to(torch.long)
        .tolist()
    )

    print("=" * 70)
    print(f"Inspection: {token_file.name}")
    print(f"Shape:      {tuple(token_tensor.shape)}")
    print(f"Data type:  {token_tensor.dtype}")
    print()

    print("Decoded beginning:")
    print(
        tokenizer.decode(
            beginning_ids,
            skip_special_tokens=False,
        )
    )

    print()
    print("Decoded ending:")
    print(
        tokenizer.decode(
            ending_ids,
            skip_special_tokens=False,
        )
    )


def main() -> None:
    tokenizer = load_tokenizer()

    print(
        f"Tokenizer vocabulary size: "
        f"{tokenizer.get_vocab_size():,}"
    )
    print()

    dataset_files = {
        "wikitext_train": (
            WIKITEXT_DIRECTORY / "train.txt",
            OUTPUT_DIRECTORY / "wikitext_train.pt",
        ),
        "wikitext_validation": (
            WIKITEXT_DIRECTORY / "validation.txt",
            OUTPUT_DIRECTORY / "wikitext_validation.pt",
        ),
        "wikitext_test": (
            WIKITEXT_DIRECTORY / "test.txt",
            OUTPUT_DIRECTORY / "wikitext_test.pt",
        ),
        "allen_train": (
            ALLEN_FILE,
            OUTPUT_DIRECTORY / "allen_train.pt",
        ),
    }

    statistics: dict[str, object] = {
        "tokenizer_file": str(
            TOKENIZER_FILE.relative_to(PROJECT_ROOT)
        ),
        "vocabulary_size": tokenizer.get_vocab_size(),
        "token_dtype": "torch.int32",
        "datasets": {},
    }

    for dataset_name, (
        input_file,
        output_file,
    ) in dataset_files.items():
        print("=" * 70)
        print(f"Tokenizing {dataset_name}")
        print(f"Input:  {input_file}")
        print(f"Output: {output_file}")
        print()

        dataset_statistics = tokenize_file(
            tokenizer=tokenizer,
            input_file=input_file,
            output_file=output_file,
        )

        statistics["datasets"][dataset_name] = dataset_statistics

        print()
        print(
            f"Completed {dataset_name}: "
            f"{dataset_statistics['token_count']:,} tokens"
        )

    statistics_file = (
        OUTPUT_DIRECTORY
        / "tokenization_stats.json"
    )

    statistics_file.write_text(
        json.dumps(
            statistics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 70)
    print(f"Statistics saved to: {statistics_file}")

    inspect_saved_tokens(
        tokenizer,
        OUTPUT_DIRECTORY / "wikitext_train.pt",
    )

    inspect_saved_tokens(
        tokenizer,
        OUTPUT_DIRECTORY / "allen_train.pt",
    )


if __name__ == "__main__":
    main()