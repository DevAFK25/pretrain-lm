from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WIKITEXT_TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wikitext-103"
    / "train.txt"
)

HOMEOPATHY_TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "homeopathy"
    / "allens_encyclopedia_i_x_clean.txt"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "tokenizer"
    / "output"
    / "tokenizer.json"
)


# ---------------------------------------------------------------------------
# Tokenizer configuration
# ---------------------------------------------------------------------------

VOCAB_SIZE = 24_000
MIN_FREQUENCY = 2

END_OF_TEXT_TOKEN = "<|endoftext|>"
REMEDY_TOKEN = "<REMEDY>"
SECTION_TOKEN = "<SECTION>"


def get_training_files(include_homeopathy: bool) -> list[Path]:
    """
    Return the corpus files used to train the tokenizer.

    WikiText is always included. The homeopathy sample can be enabled once
    the domain corpus is available.
    """
    training_files = [WIKITEXT_TRAIN_FILE]

    if include_homeopathy:
        training_files.append(HOMEOPATHY_TRAIN_FILE)

    return training_files


def validate_training_files(training_files: list[Path]) -> None:
    """Raise a clear error if any required training file is missing."""

    missing_files = [
        file_path
        for file_path in training_files
        if not file_path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(
            f"  - {file_path}"
            for file_path in missing_files
        )

        raise FileNotFoundError(
            "The following tokenizer training files were not found:\n"
            f"{missing_text}"
        )


def build_tokenizer() -> Tokenizer:
    """
    Create an untrained byte-level BPE tokenizer.

    The tokenizer begins with no learned merges. The trainer will learn its
    vocabulary from the supplied corpus files.
    """
    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = decoders.ByteLevel()

    return tokenizer


def build_trainer() -> trainers.BpeTrainer:
    """Create the BPE trainer and define its vocabulary policy."""

    return trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=[
            END_OF_TEXT_TOKEN,
            REMEDY_TOKEN,
            SECTION_TOKEN,
        ],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )


def train_tokenizer(training_files: list[Path]) -> Tokenizer:
    """Train the tokenizer on one or more corpus files."""

    tokenizer = build_tokenizer()
    trainer = build_trainer()

    tokenizer.train(
        files=[str(file_path) for file_path in training_files],
        trainer=trainer,
    )

    return tokenizer


def inspect_tokenizer(tokenizer: Tokenizer) -> None:
    """Inspect tokenization and verify encode/decode behaviour."""

    test_sentences = [
        "The patient feels better after rest.",
        "Bryonia improves pain after rest.",
        "Belladonna is described in the Materia Medica.",

        # Historical and medical spellings
        "diarrhœa",
        "hæmorrhoids",
        "fæces",
        "œsophagus",
        "hæmorrhage",
        "fétide",
        "nausea",
        "constipation",

        # Remedy names
        "Bryonia",
        "Belladonna",
        "Aconitum",
        "Abies canadensis",
        "Nux vomica",

        # Structural tokens
        "<REMEDY> Abies canadensis [Abies-c]",
        "<SECTION> Mind.",
        "First remedy.<|endoftext|><REMEDY> Second remedy [Test.]",
        ]

    print(f"\nVocabulary size: {tokenizer.get_vocab_size()}")
    print(
        "End-of-text token ID:",
        tokenizer.token_to_id(END_OF_TEXT_TOKEN),
    )

    for sentence in test_sentences:
        encoding = tokenizer.encode(sentence)
        
        print("\n" + "=" * 70)
        print(f"Original: {sentence}")
        print(f"Tokens:   {encoding.tokens}")
        print(f"Token count: {len(encoding.ids)}")
        print(f"IDs:      {encoding.ids}")
        print(f"Decoded:  {tokenizer.decode(encoding.ids, skip_special_tokens=False)}")


def main() -> None:
    # Keep this False until the homeopathy tokenizer sample is available. NOW SET TO TRUE
    include_homeopathy = True

    training_files = get_training_files(
        include_homeopathy=include_homeopathy
    )

    validate_training_files(training_files)

    print("Training tokenizer using:")

    for file_path in training_files:
        print(f"  - {file_path}")

    print(f"\nTarget vocabulary size: {VOCAB_SIZE:,}")
    print(f"Minimum token frequency: {MIN_FREQUENCY}")

    tokenizer = train_tokenizer(training_files)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(str(OUTPUT_FILE))

    print(f"\nTokenizer saved to: {OUTPUT_FILE}")

    for special_token in [END_OF_TEXT_TOKEN, REMEDY_TOKEN, SECTION_TOKEN,]:
        print(
            f"{special_token} token ID:",
            tokenizer.token_to_id(special_token),
        )

    inspect_tokenizer(tokenizer)


if __name__ == "__main__":
    main()