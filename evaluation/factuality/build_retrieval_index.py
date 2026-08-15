from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CORPUS_DIR = (
    PROJECT_ROOT
    / "Data2"
    / "structured"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "factuality"
    / "cache"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "retrieval_passages.json"
)


MAX_WORDS_PER_PASSAGE = 220
OVERLAP_WORDS = 40


def normalize_remedy_name(name: str) -> str:
    name = name.strip()

    # Remove abbreviation suffix:
    # "Bryonia alba [Bry.]" -> "Bryonia alba"
    name = re.sub(
        r"\s*\[[^\]]+\]\s*$",
        "",
        name,
    )

    return name.strip()


def chunk_words(
    text: str,
    max_words: int,
    overlap_words: int,
) -> list[str]:

    words = text.split()

    if not words:
        return []

    if len(words) <= max_words:
        return [" ".join(words)]

    chunks = []
    step = max_words - overlap_words

    start = 0

    while start < len(words):
        end = min(
            start + max_words,
            len(words),
        )

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start += step

    return chunks


def parse_structured_file(
    path: Path,
) -> list[dict]:

    text = path.read_text(
        encoding="utf-8"
    )

    lines = text.splitlines()

    passages = []

    current_remedy = None
    current_section = None
    buffer = []

    def flush():
        nonlocal buffer

        if not buffer:
            return

        raw_text = "\n".join(buffer).strip()

        if not raw_text:
            buffer = []
            return

        chunks = chunk_words(
            raw_text,
            MAX_WORDS_PER_PASSAGE,
            OVERLAP_WORDS,
        )

        for chunk in chunks:
            passages.append(
                {
                    "source_file": path.name,
                    "remedy": current_remedy,
                    "section": current_section,
                    "text": chunk,
                }
            )

        buffer = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("<REMEDY>"):
            flush()

            current_remedy = normalize_remedy_name(
                stripped.replace(
                    "<REMEDY>",
                    "",
                    1,
                )
            )

            current_section = None
            continue

        if stripped.startswith("<SECTION>"):
            flush()

            current_section = (
                stripped.replace(
                    "<SECTION>",
                    "",
                    1,
                ).strip()
            )

            continue

        buffer.append(line)

    flush()

    return passages


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        CORPUS_DIR.glob("*.txt")
    )

    if not files:
        raise RuntimeError(
            f"No .txt files found in {CORPUS_DIR}"
        )

    all_passages = []

    for path in files:
        passages = parse_structured_file(path)

        all_passages.extend(passages)

        remedies = {
            p["remedy"]
            for p in passages
            if p["remedy"]
        }

        print(
            f"{path.name}: "
            f"{len(passages):,} passages | "
            f"{len(remedies):,} remedies"
        )

    for index, passage in enumerate(
        all_passages,
        start=1,
    ):
        passage["passage_id"] = (
            f"passage_{index:06d}"
        )

    OUTPUT_FILE.write_text(
        json.dumps(
            all_passages,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    remedy_count = len(
        {
            p["remedy"]
            for p in all_passages
            if p["remedy"]
        }
    )

    print()
    print("=" * 80)
    print("RETRIEVAL INDEX BUILD COMPLETE")
    print("=" * 80)
    print(
        f"Passages: {len(all_passages):,}"
    )
    print(
        f"Unique remedies: {remedy_count:,}"
    )
    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
