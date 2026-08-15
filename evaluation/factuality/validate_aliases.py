from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .remedy_aliases import (
    REMEDY_ALIASES,
    normalize_name,
    remedy_matches,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

PASSAGES_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "factuality"
    / "cache"
    / "retrieval_passages.json"
)


def main():

    passages = json.loads(
        PASSAGES_FILE.read_text(
            encoding="utf-8"
        )
    )

    corpus_counts = Counter(
        normalize_name(p["remedy"])
        for p in passages
        if p.get("remedy")
    )

    print("=" * 80)
    print("REMEDY ALIAS VALIDATION")
    print("=" * 80)

    for canonical, aliases in (
        REMEDY_ALIASES.items()
    ):

        matched = [
            p
            for p in passages
            if remedy_matches(
                canonical,
                p.get("remedy"),
            )
        ]

        print()
        print(
            canonical.upper()
        )

        print(
            "Allowed headings:"
        )

        total_expected = 0

        for alias in sorted(aliases):

            normalized = normalize_name(
                alias
            )

            count = corpus_counts.get(
                normalized,
                0,
            )

            total_expected += count

            print(
                f"  {alias:<30} "
                f"{count:>5} passages"
            )

        print(
            f"TOTAL MATCHED: "
            f"{len(matched)} passages"
        )

        if len(matched) != total_expected:
            raise RuntimeError(
                f"Alias validation failed "
                f"for {canonical}"
            )

        if len(matched) < 5:
            raise RuntimeError(
                f"Too few passages for "
                f"{canonical}: {len(matched)}"
            )

    print()
    print("=" * 80)
    print("NEGATIVE MATCH TESTS")
    print("=" * 80)

    negative_tests = [
        (
            "sulphur",
            "Natrum sulphuricum",
        ),
        (
            "sulphur",
            "Kali sulphuricum",
        ),
        (
            "sulphur",
            "Calcarea sulphurica",
        ),
        (
            "aconitum",
            "Aconitum ferox",
        ),
        (
            "aconitum",
            "Aconitum cammarum",
        ),
        (
            "pulsatilla",
            "Pulsatilla nuttalliana",
        ),
        (
            "rhus toxicodendron",
            "Rhus vernix",
        ),
    ]

    for prompt_remedy, corpus_remedy in (
        negative_tests
    ):

        result = remedy_matches(
            prompt_remedy,
            corpus_remedy,
        )

        print(
            f"{prompt_remedy:<20} "
            f"X "
            f"{corpus_remedy:<30} "
            f"{'PASS' if not result else 'FAIL'}"
        )

        if result:
            raise RuntimeError(
                "Invalid cross-remedy match: "
                f"{prompt_remedy} -> "
                f"{corpus_remedy}"
            )

    print()
    print("=" * 80)
    print("ALL REMEDY ALIASES VALIDATED")
    print("=" * 80)


if __name__ == "__main__":
    main()
