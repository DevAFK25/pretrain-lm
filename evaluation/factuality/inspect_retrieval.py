from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import requests

from .remedy_aliases import (
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

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "factuality"
    / "cache"
    / "retrieval_passage_embeddings.npy"
)

CLAIMS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "factuality"
    / "outputs"
    / "extracted_claims.json"
)

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

EMBEDDING_MODEL = (
    "text-embedding-nomic-embed-text-v1.5"
)

TOP_K = 5


def extract_remedy(
    prompt: str,
) -> str:

    return (
        prompt
        .replace("<REMEDY>", "", 1)
        .strip()
    )


def embed_query(
    text: str,
) -> np.ndarray:

    response = requests.post(
        f"{LM_STUDIO_BASE_URL}/embeddings",
        json={
            "model": EMBEDDING_MODEL,
            "input": (
                f"search_query: {text}"
            ),
        },
        timeout=180,
    )

    response.raise_for_status()

    vector = np.asarray(
        response.json()[
            "data"
        ][0]["embedding"],
        dtype=np.float32,
    )

    norm = np.linalg.norm(vector)

    return (
        vector / norm
        if norm > 0
        else vector
    )


def main():

    passages = json.loads(
        PASSAGES_FILE.read_text(
            encoding="utf-8"
        )
    )

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    claims = json.loads(
        CLAIMS_FILE.read_text(
            encoding="utf-8"
        )
    )

    # Fixed spread through the claim list:
    # no cherry-picking based on retrieval result.
    indices = [
        0,
        8,
        16,
        24,
        40,
        64,
        96,
        128,
        160,
        187,
    ]

    for claim_index in indices:

        item = claims[claim_index]

        remedy = extract_remedy(
            item["prompt"]
        )

        candidate_indices = [
            i
            for i, passage
            in enumerate(passages)
            if remedy_matches(
                remedy,
                passage.get("remedy"),
            )
        ]

        if not candidate_indices:
            raise RuntimeError(
                f"No passages for {remedy}"
            )

        query = embed_query(
            item["claim"]
        )

        candidate_matrix = embeddings[
            candidate_indices
        ]

        similarities = (
            candidate_matrix
            @ query
        )

        order = np.argsort(
            similarities
        )[-TOP_K:][::-1]

        print()
        print("=" * 100)
        print(
            f"CLAIM: {item['claim']}"
        )
        print(
            f"CHECKPOINT: "
            f"{item['checkpoint']}"
        )
        print(
            f"REMEDY: {remedy}"
        )
        print(
            f"CANDIDATE PASSAGES: "
            f"{len(candidate_indices)}"
        )
        print("=" * 100)

        for rank, local_index in enumerate(
            order,
            start=1,
        ):

            global_index = (
                candidate_indices[
                    int(local_index)
                ]
            )

            passage = passages[
                global_index
            ]

            similarity = float(
                similarities[
                    local_index
                ]
            )

            print()
            print(
                f"[{rank}] "
                f"similarity={similarity:.4f}"
            )

            print(
                f"source:  "
                f"{passage['source_file']}"
            )

            print(
                f"remedy:  "
                f"{passage['remedy']}"
            )

            print(
                f"section: "
                f"{passage['section']}"
            )

            print()
            print(
                passage["text"]
            )


if __name__ == "__main__":
    main()
