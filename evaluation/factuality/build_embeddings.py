from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import requests


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

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

EMBEDDING_MODEL = (
    "text-embedding-nomic-embed-text-v1.5"
)

BATCH_SIZE = 16
TIMEOUT = 180


def normalize_rows(
    matrix: np.ndarray,
) -> np.ndarray:

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1.0

    return matrix / norms


def embed_batch(
    texts: list[str],
) -> np.ndarray:

    prepared = [
        f"search_document: {text}"
        for text in texts
    ]

    response = requests.post(
        f"{LM_STUDIO_BASE_URL}/embeddings",
        json={
            "model": EMBEDDING_MODEL,
            "input": prepared,
        },
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    vectors = np.asarray(
        [
            item["embedding"]
            for item
            in response.json()["data"]
        ],
        dtype=np.float32,
    )

    return normalize_rows(vectors)


def main():

    passages = json.loads(
        PASSAGES_FILE.read_text(
            encoding="utf-8"
        )
    )

    print(
        f"Passages: {len(passages):,}"
    )

    if EMBEDDINGS_FILE.exists():
        existing = np.load(
            EMBEDDINGS_FILE
        )

        if existing.shape[0] == len(
            passages
        ):
            print(
                "Embeddings already exist:"
            )
            print(
                EMBEDDINGS_FILE
            )
            print(
                f"Shape: {existing.shape}"
            )
            return

    batches = math.ceil(
        len(passages)
        / BATCH_SIZE
    )

    matrices = []

    print(
        f"Embedding with "
        f"{EMBEDDING_MODEL}..."
    )

    for batch_number, start in enumerate(
        range(
            0,
            len(passages),
            BATCH_SIZE,
        ),
        start=1,
    ):

        batch = passages[
            start:
            start + BATCH_SIZE
        ]

        matrix = embed_batch(
            [
                item["text"]
                for item in batch
            ]
        )

        matrices.append(matrix)

        if (
            batch_number == 1
            or batch_number % 50 == 0
            or batch_number == batches
        ):
            done = min(
                start + BATCH_SIZE,
                len(passages),
            )

            print(
                f"  {batch_number:4d}/"
                f"{batches:4d} batches | "
                f"{done:,}/"
                f"{len(passages):,} passages"
            )

    embeddings = np.concatenate(
        matrices,
        axis=0,
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings,
    )

    print()
    print("=" * 80)
    print("PASSAGE EMBEDDINGS COMPLETE")
    print("=" * 80)
    print(
        f"Shape: {embeddings.shape}"
    )
    print(
        f"Saved to:\n{EMBEDDINGS_FILE}"
    )


if __name__ == "__main__":
    main()
