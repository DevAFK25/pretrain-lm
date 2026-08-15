from __future__ import annotations

import re


# ============================================================
# CANONICAL REMEDY ALIASES
#
# Keys = remedy names used in our evaluation prompts.
# Values = exact remedy headings in the structured corpus that
# are allowed to count as evidence for that remedy.
#
# IMPORTANT:
# These are intentionally conservative.
# Related remedies/species are NOT automatically included.
# ============================================================

REMEDY_ALIASES = {
    "bryonia": {
        "bryonia",
        "bryonia alba",
    },

    "nux vomica": {
        "nux vomica",
    },

    "pulsatilla": {
        "pulsatilla",
        "pulsatilla nigricans",
    },

    "sulphur": {
        "sulphur",
    },

    "aconitum": {
        "aconitum napellus",
    },

    "belladonna": {
        "belladonna",
    },

    "arnica": {
        "arnica",
        "arnica montana",
    },

    "rhus toxicodendron": {
        "rhus toxicodendron",
    },
}


def normalize_name(name: str) -> str:
    """
    Normalize remedy names for exact comparison.

    This does NOT perform fuzzy matching.
    """

    name = name.strip().lower()

    # Collapse repeated whitespace.
    name = re.sub(
        r"\s+",
        " ",
        name,
    )

    # Remove trailing punctuation only.
    name = name.rstrip(
        " .,:;"
    )

    return name


def get_allowed_aliases(
    prompt_remedy: str,
) -> set[str]:
    """
    Return exact corpus remedy headings allowed for the
    requested evaluation remedy.
    """

    canonical = normalize_name(
        prompt_remedy
    )

    if canonical not in REMEDY_ALIASES:
        raise KeyError(
            "No remedy alias mapping defined for: "
            f"{prompt_remedy!r}"
        )

    return {
        normalize_name(alias)
        for alias in REMEDY_ALIASES[
            canonical
        ]
    }


def remedy_matches(
    prompt_remedy: str,
    corpus_remedy: str | None,
) -> bool:
    """
    True only when corpus_remedy is explicitly permitted
    for prompt_remedy.
    """

    if corpus_remedy is None:
        return False

    allowed = get_allowed_aliases(
        prompt_remedy
    )

    corpus_name = normalize_name(
        corpus_remedy
    )

    return corpus_name in allowed
