from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


WORD_PATTERN = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)

# Characters already observed in the corpus or commonly produced by OCR/export.
WATCHED_CHARACTERS = {
    "∫": "integral-style formatting marker",
    "ƒ": "florin-style formatting marker",
    "œ": "oe ligature",
    "æ": "ae ligature",
    "�": "Unicode replacement character",
    "\t": "tab",
    "\r": "carriage return",
    "\u00a0": "non-breaking space",
    "\u200b": "zero-width space",
    "\ufeff": "byte-order mark",
}


def read_text_safely(path: Path) -> tuple[str, str]:
    """
    Read a text file without modifying it.

    UTF-8 is attempted first. Other encodings are fallbacks for older
    text exports.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Unable to decode {path}",
    )


def sha256_bytes(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def normalized_text_hash(text: str) -> str:
    """
    Used only for duplicate detection.

    This does not alter the source file. It ignores differences in
    casing and whitespace.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def count_control_characters(text: str) -> Counter[str]:
    controls: Counter[str] = Counter()

    for char in text:
        category = unicodedata.category(char)

        # Keep normal line separators and tabs out of the generic count.
        if category.startswith("C") and char not in {"\n", "\r", "\t"}:
            controls[f"U+{ord(char):04X}"] += 1

    return controls


def count_non_ascii_characters(text: str) -> Counter[str]:
    return Counter(char for char in text if ord(char) > 127)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only corpus audit for Data2."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data2"),
        help="Directory containing .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reports/data2"),
        help="Directory in which audit reports will be written.",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir.resolve()}"
        )

    txt_files = sorted(data_dir.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found inside: {data_dir.resolve()}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    file_rows: list[dict] = []
    watched_rows: list[dict] = []
    non_ascii_total: Counter[str] = Counter()
    control_total: Counter[str] = Counter()

    exact_hash_groups: defaultdict[str, list[str]] = defaultdict(list)
    normalized_hash_groups: defaultdict[str, list[str]] = defaultdict(list)

    corpus_totals = {
        "files": 0,
        "bytes": 0,
        "characters": 0,
        "words": 0,
        "lines": 0,
        "nonempty_lines": 0,
        "blank_lines": 0,
        "paragraphs": 0,
    }

    for path in txt_files:
        text, encoding = read_text_safely(path)

        lines = text.splitlines()
        nonempty_lines = [line for line in lines if line.strip()]
        blank_lines = len(lines) - len(nonempty_lines)

        paragraph_blocks = [
            block
            for block in re.split(r"\n\s*\n", text)
            if block.strip()
        ]

        words = WORD_PATTERN.findall(text)
        line_lengths = [len(line) for line in lines]

        watched_counts = {
            character: text.count(character)
            for character in WATCHED_CHARACTERS
        }

        non_ascii = count_non_ascii_characters(text)
        controls = count_control_characters(text)

        non_ascii_total.update(non_ascii)
        control_total.update(controls)

        exact_hash = sha256_bytes(path)
        normalized_hash = normalized_text_hash(text)

        exact_hash_groups[exact_hash].append(path.name)
        normalized_hash_groups[normalized_hash].append(path.name)

        row = {
            "filename": path.name,
            "encoding_used": encoding,
            "size_bytes": path.stat().st_size,
            "characters": len(text),
            "words": len(words),
            "lines": len(lines),
            "nonempty_lines": len(nonempty_lines),
            "blank_lines": blank_lines,
            "paragraph_blocks": len(paragraph_blocks),
            "average_line_length": (
                round(sum(line_lengths) / len(line_lengths), 2)
                if line_lengths
                else 0
            ),
            "maximum_line_length": max(line_lengths, default=0),
            "lines_over_500_chars": sum(
                length > 500 for length in line_lengths
            ),
            "lines_over_2000_chars": sum(
                length > 2000 for length in line_lengths
            ),
            "zz_book_count": len(
                re.findall(r"\[\s*zz-book\s*\]", text, flags=re.IGNORECASE)
            ),
            "hash_sha256": exact_hash,
            "normalized_text_hash": normalized_hash,
        }

        for character, description in WATCHED_CHARACTERS.items():
            row[f"count_{ord(character):04X}"] = watched_counts[character]

            if watched_counts[character] > 0:
                watched_rows.append(
                    {
                        "filename": path.name,
                        "character": repr(character),
                        "unicode_codepoint": f"U+{ord(character):04X}",
                        "description": description,
                        "count": watched_counts[character],
                    }
                )

        file_rows.append(row)

        corpus_totals["files"] += 1
        corpus_totals["bytes"] += path.stat().st_size
        corpus_totals["characters"] += len(text)
        corpus_totals["words"] += len(words)
        corpus_totals["lines"] += len(lines)
        corpus_totals["nonempty_lines"] += len(nonempty_lines)
        corpus_totals["blank_lines"] += blank_lines
        corpus_totals["paragraphs"] += len(paragraph_blocks)

    duplicate_rows: list[dict] = []

    for hash_type, groups in [
        ("exact_file", exact_hash_groups),
        ("normalized_text", normalized_hash_groups),
    ]:
        for digest, filenames in groups.items():
            if len(filenames) > 1:
                duplicate_rows.append(
                    {
                        "duplicate_type": hash_type,
                        "hash": digest,
                        "file_count": len(filenames),
                        "filenames": " | ".join(filenames),
                    }
                )

    non_ascii_rows = []

    for character, count in non_ascii_total.most_common():
        non_ascii_rows.append(
            {
                "character": repr(character),
                "unicode_codepoint": f"U+{ord(character):04X}",
                "unicode_name": unicodedata.name(character, "UNKNOWN"),
                "count": count,
            }
        )

    control_rows = [
        {
            "unicode_codepoint": codepoint,
            "count": count,
        }
        for codepoint, count in control_total.most_common()
    ]

    file_stats_path = output_dir / "file_stats.csv"
    with file_stats_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=file_rows[0].keys())
        writer.writeheader()
        writer.writerows(file_rows)

    watched_path = output_dir / "watched_characters.csv"
    with watched_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "filename",
            "character",
            "unicode_codepoint",
            "description",
            "count",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(watched_rows)

    non_ascii_path = output_dir / "non_ascii_characters.csv"
    with non_ascii_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "character",
            "unicode_codepoint",
            "unicode_name",
            "count",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(non_ascii_rows)

    controls_path = output_dir / "control_characters.csv"
    with controls_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["unicode_codepoint", "count"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(control_rows)

    duplicates_path = output_dir / "duplicate_files.csv"
    with duplicates_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "duplicate_type",
            "hash",
            "file_count",
            "filenames",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(duplicate_rows)

    largest_files = sorted(
        file_rows,
        key=lambda row: row["characters"],
        reverse=True,
    )[:10]

    longest_line_files = sorted(
        file_rows,
        key=lambda row: row["maximum_line_length"],
        reverse=True,
    )[:10]

    summary = {
        "data_directory": str(data_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "totals": corpus_totals,
        "estimated_megabytes": round(
            corpus_totals["bytes"] / (1024 * 1024),
            3,
        ),
        "exact_duplicate_groups": sum(
            len(files) > 1 for files in exact_hash_groups.values()
        ),
        "normalized_duplicate_groups": sum(
            len(files) > 1 for files in normalized_hash_groups.values()
        ),
        "largest_files_by_character_count": [
            {
                "filename": row["filename"],
                "characters": row["characters"],
                "words": row["words"],
            }
            for row in largest_files
        ],
        "files_with_longest_lines": [
            {
                "filename": row["filename"],
                "maximum_line_length": row["maximum_line_length"],
                "lines_over_2000_chars": row["lines_over_2000_chars"],
            }
            for row in longest_line_files
        ],
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== DATA2 AUDIT COMPLETE =====")
    print(f"Files:          {corpus_totals['files']:,}")
    print(f"Size:           {summary['estimated_megabytes']:,} MB")
    print(f"Characters:     {corpus_totals['characters']:,}")
    print(f"Words:          {corpus_totals['words']:,}")
    print(f"Lines:          {corpus_totals['lines']:,}")
    print(f"Paragraphs:     {corpus_totals['paragraphs']:,}")
    print(
        "Exact duplicate groups:",
        summary["exact_duplicate_groups"],
    )
    print(
        "Normalized duplicate groups:",
        summary["normalized_duplicate_groups"],
    )
    print(f"\nReports saved to: {output_dir.resolve()}")
    print("No source files were modified.")


if __name__ == "__main__":
    main()