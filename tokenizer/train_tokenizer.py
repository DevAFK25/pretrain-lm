from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


# Paths are resolved relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_FILE = PROJECT_ROOT / "data" / "debug" / "sample.txt"
OUTPUT_FILE = PROJECT_ROOT / "tokenizer" / "debug_tokenizer.json"


def train_tokenizer() -> Tokenizer:
    """Train a byte-level BPE tokenizer on the debug corpus."""

    # The BPE model begins without a learned vocabulary.
    # <unk> is reserved for text the tokenizer cannot represent.
    tokenizer = Tokenizer(
        models.BPE(unk_token="<unk>")
    )

    # Split text at byte-level boundaries before BPE learns its merges.
    # This preserves punctuation, whitespace, and any UTF-8 text.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    # Convert byte-level tokens back into normal readable text.
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=300,
        min_frequency=2,
        special_tokens=[
            "<pad>",
            "<unk>",
            "<bos>",
            "<eos>",
        ],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train(
        files=[str(TRAINING_FILE)],
        trainer=trainer,
    )

    return tokenizer


def inspect_tokenizer(tokenizer: Tokenizer) -> None:
    """Encode and decode a few sentences so we can inspect the result."""

    test_sentences = [
        "The patient feels better after rest.",
        "The pain becomes worse after movement.",
        "Bryonia improves pain after rest.",
    ]

    print(f"\nVocabulary size: {tokenizer.get_vocab_size()}")

    for sentence in test_sentences:
        encoding = tokenizer.encode(sentence)

        print("\n" + "=" * 60)
        print(f"Original: {sentence}")
        print(f"Tokens:   {encoding.tokens}")
        print(f"IDs:      {encoding.ids}")
        print(f"Decoded:  {tokenizer.decode(encoding.ids)}")


def main() -> None:
    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAINING_FILE}"
        )

    tokenizer = train_tokenizer()
    tokenizer.save(str(OUTPUT_FILE))

    print(f"Tokenizer saved to: {OUTPUT_FILE}")
    inspect_tokenizer(tokenizer)


if __name__ == "__main__":
    main()