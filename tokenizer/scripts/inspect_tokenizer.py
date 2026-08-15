from pathlib import Path

from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOKENIZER_FILE = (
    PROJECT_ROOT
    / "tokenizer"
    / "output"
    / "tokenizer.json"
)

TEST_TEXTS = [
    "diarrhœa",
    "hæmorrhoids",
    "fæces",
    "œsophagus",
    "hæmorrhage",
    "fétide",
    "nausea",
    "constipation",
    "Bryonia",
    "Belladonna",
    "Aconitum",
    "Abies canadensis",
    "Nux vomica",
    "<REMEDY> Abies canadensis [Abies-c]",
    "<SECTION> Mind.",
]


def main() -> None:
    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_FILE}"
        )

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )

    print(
        f"Vocabulary size: {tokenizer.get_vocab_size():,}"
    )

    for text in TEST_TEXTS:
        encoding = tokenizer.encode(text)

        print("\n" + "=" * 70)
        print(f"Original:    {text}")
        print(f"Tokens:      {encoding.tokens}")
        print(f"Token count: {len(encoding.ids)}")
        print(f"IDs:         {encoding.ids}")
        print(
            "Decoded:     "
            f"{tokenizer.decode(
                encoding.ids,
                skip_special_tokens=False,
            )}"
        )


if __name__ == "__main__":
    main()