from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .remedy_aliases import remedy_matches


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = (
    "MoritzLaurer/"
    "DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

CLAIMS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "factuality"
    / "outputs"
    / "extracted_claims.json"
)

PASSAGES_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "factuality"
    / "cache"
    / "retrieval_passages.json"
)

PASSAGE_EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "factuality"
    / "cache"
    / "retrieval_passage_embeddings.npy"
)

CLAIM_EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "factuality"
    / "cache"
    / "claim_embeddings.npy"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "factuality"
    / "outputs"
)

DETAILED_RESULTS_FILE = (
    OUTPUT_DIR
    / "nli_v2_claim_results.json"
)

SUMMARY_JSON_FILE = (
    OUTPUT_DIR
    / "nli_v2_summary.json"
)

SUMMARY_CSV_FILE = (
    OUTPUT_DIR
    / "nli_v2_summary.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_K_EVIDENCE = 5

MAX_LENGTH = 512

NLI_BATCH_SIZE = 8

MIN_DECISIVE_PROBABILITY = 0.50
MIN_DECISIVE_MARGIN = 0.15


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def save_json(path: Path, data):
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def extract_remedy(
    prompt: str,
) -> str:

    if not prompt.startswith("<REMEDY>"):
        raise ValueError(
            f"Expected remedy prompt, got: "
            f"{prompt!r}"
        )

    return (
        prompt
        .replace("<REMEDY>", "", 1)
        .strip()
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    required = [
        CLAIMS_FILE,
        PASSAGES_FILE,
        PASSAGE_EMBEDDINGS_FILE,
        CLAIM_EMBEDDINGS_FILE,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )

    claims = load_json(
        CLAIMS_FILE
    )

    passages = load_json(
        PASSAGES_FILE
    )

    passage_embeddings = np.load(
        PASSAGE_EMBEDDINGS_FILE
    )

    claim_embeddings = np.load(
        CLAIM_EMBEDDINGS_FILE
    )

    if len(passages) != passage_embeddings.shape[0]:
        raise RuntimeError(
            "Passage count does not match "
            "passage embedding count."
        )

    if len(claims) != claim_embeddings.shape[0]:
        raise RuntimeError(
            "Claim count does not match "
            "claim embedding count."
        )

    print(
        f"Claims:              {len(claims):,}"
    )
    print(
        f"Passages:            {len(passages):,}"
    )
    print(
        f"Passage embeddings:  "
        f"{passage_embeddings.shape}"
    )
    print(
        f"Claim embeddings:    "
        f"{claim_embeddings.shape}"
    )
    print()

    return (
        claims,
        passages,
        passage_embeddings,
        claim_embeddings,
    )


# ============================================================
# HARD-FILTERED RETRIEVAL
# ============================================================

def retrieve_top_k(
    claim_embedding: np.ndarray,
    prompt: str,
    passages: list[dict],
    passage_embeddings: np.ndarray,
) -> list[dict]:

    remedy = extract_remedy(
        prompt
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
            f"No candidate passages for "
            f"remedy: {remedy}"
        )

    candidate_matrix = (
        passage_embeddings[
            candidate_indices
        ]
    )

    similarities = (
        candidate_matrix
        @ claim_embedding
    )

    top_local_indices = np.argsort(
        similarities
    )[
        -TOP_K_EVIDENCE:
    ][::-1]

    results = []

    for rank, local_index in enumerate(
        top_local_indices,
        start=1,
    ):

        local_index = int(
            local_index
        )

        global_index = (
            candidate_indices[
                local_index
            ]
        )

        passage = passages[
            global_index
        ]

        results.append(
            {
                "rank": rank,
                "passage_id": (
                    passage["passage_id"]
                ),
                "source_file": (
                    passage["source_file"]
                ),
                "remedy": (
                    passage["remedy"]
                ),
                "section": (
                    passage["section"]
                ),
                "similarity": float(
                    similarities[
                        local_index
                    ]
                ),
                "text": (
                    passage["text"]
                ),
            }
        )

    return results


# ============================================================
# NLI MODEL
# ============================================================

def load_nli_model():

    device = get_device()

    print(
        f"NLI device: {device}"
    )
    print(
        f"NLI model:  {MODEL_NAME}"
    )
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME
        )
        .to(device)
    )

    model.eval()

    return (
        tokenizer,
        model,
        device,
    )


@torch.no_grad()
def run_nli_batch(
    tokenizer,
    model,
    device,
    premises: list[str],
    hypotheses: list[str],
) -> list[dict]:

    encoded = tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(device)
        for key, value
        in encoded.items()
    }

    logits = model(
        **encoded
    ).logits

    probs = torch.softmax(
        logits,
        dim=-1,
    ).cpu().numpy()

    # Model-card label order:
    # entailment, neutral, contradiction

    results = []

    for row in probs:
        results.append(
            {
                "entailment": float(
                    row[0]
                ),
                "neutral": float(
                    row[1]
                ),
                "contradiction": float(
                    row[2]
                ),
            }
        )

    return results


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_claim(
    evidence_results: list[dict],
) -> dict:

    strongest_entailment = max(
        evidence_results,
        key=lambda x: (
            x["nli"]["entailment"]
        ),
    )

    strongest_contradiction = max(
        evidence_results,
        key=lambda x: (
            x["nli"]["contradiction"]
        ),
    )

    max_entailment = (
        strongest_entailment[
            "nli"
        ]["entailment"]
    )

    max_contradiction = (
        strongest_contradiction[
            "nli"
        ]["contradiction"]
    )

    entailment_margin = (
        max_entailment
        - max_contradiction
    )

    contradiction_margin = (
        max_contradiction
        - max_entailment
    )

    entailment_scores = sorted(
        [
            item["nli"]["entailment"]
            for item in evidence_results
        ],
        reverse=True,
    )

    contradiction_scores = sorted(
        [
            item["nli"]["contradiction"]
            for item in evidence_results
        ],
        reverse=True,
    )

    top2_entailment_mean = float(
        np.mean(
            entailment_scores[:2]
        )
    )

    top2_contradiction_mean = float(
        np.mean(
            contradiction_scores[:2]
        )
    )

    # Keep same conservative rule as v1
    # so changes reflect retrieval quality,
    # not threshold changes.

    if (
        max_entailment
        >= MIN_DECISIVE_PROBABILITY
        and entailment_margin
        >= MIN_DECISIVE_MARGIN
    ):
        verdict = "SUPPORTED"

    elif (
        max_contradiction
        >= MIN_DECISIVE_PROBABILITY
        and contradiction_margin
        >= MIN_DECISIVE_MARGIN
    ):
        verdict = "CONTRADICTED"

    else:
        verdict = (
            "INSUFFICIENT_EVIDENCE"
        )

    return {
        "verdict": verdict,

        "max_entailment": (
            max_entailment
        ),

        "max_contradiction": (
            max_contradiction
        ),

        "entailment_margin": (
            entailment_margin
        ),

        "contradiction_margin": (
            contradiction_margin
        ),

        "top2_entailment_mean": (
            top2_entailment_mean
        ),

        "top2_contradiction_mean": (
            top2_contradiction_mean
        ),

        "strongest_supporting_evidence": {
            "rank": (
                strongest_entailment[
                    "rank"
                ]
            ),
            "passage_id": (
                strongest_entailment[
                    "passage_id"
                ]
            ),
            "source_file": (
                strongest_entailment[
                    "source_file"
                ]
            ),
            "section": (
                strongest_entailment[
                    "section"
                ]
            ),
        },

        "strongest_contradicting_evidence": {
            "rank": (
                strongest_contradiction[
                    "rank"
                ]
            ),
            "passage_id": (
                strongest_contradiction[
                    "passage_id"
                ]
            ),
            "source_file": (
                strongest_contradiction[
                    "source_file"
                ]
            ),
            "section": (
                strongest_contradiction[
                    "section"
                ]
            ),
        },
    }


# ============================================================
# FULL RUN
# ============================================================

def evaluate_all(
    claims,
    passages,
    passage_embeddings,
    claim_embeddings,
    tokenizer,
    model,
    device,
):

    results = []

    total = len(
        claims
    )

    for claim_index, item in enumerate(
        claims,
        start=1,
    ):

        evidence = retrieve_top_k(
            claim_embedding=(
                claim_embeddings[
                    claim_index - 1
                ]
            ),
            prompt=item["prompt"],
            passages=passages,
            passage_embeddings=(
                passage_embeddings
            ),
        )

        evidence_results = []

        for start in range(
            0,
            len(evidence),
            NLI_BATCH_SIZE,
        ):

            batch = evidence[
                start:
                start + NLI_BATCH_SIZE
            ]

            premises = [
                x["text"]
                for x in batch
            ]

            hypotheses = [
                item["claim"]
            ] * len(batch)

            scores = run_nli_batch(
                tokenizer=tokenizer,
                model=model,
                device=device,
                premises=premises,
                hypotheses=hypotheses,
            )

            for evidence_item, nli_scores in zip(
                batch,
                scores,
            ):
                evidence_results.append(
                    {
                        **evidence_item,
                        "nli": nli_scores,
                    }
                )

        aggregate = aggregate_claim(
            evidence_results
        )

        record = {
            "claim_id": (
                item["claim_id"]
            ),
            "sample_id": (
                item["sample_id"]
            ),
            "checkpoint": (
                item["checkpoint"]
            ),
            "prompt": (
                item["prompt"]
            ),
            "claim": (
                item["claim"]
            ),

            **aggregate,

            "evidence": (
                evidence_results
            ),
        }

        results.append(
            record
        )

        print(
            f"[{claim_index:03d}/"
            f"{total:03d}] "
            f"{item['claim_id']} | "
            f"{aggregate['verdict']:<21} | "
            f"E={aggregate['max_entailment']:.3f} "
            f"C={aggregate['max_contradiction']:.3f}"
        )

        save_json(
            DETAILED_RESULTS_FILE,
            results,
        )

    return results


# ============================================================
# SUMMARY
# ============================================================

def checkpoint_summary(
    checkpoint: str,
    results: list[dict],
) -> dict:

    subset = [
        x
        for x in results
        if x["checkpoint"]
        == checkpoint
    ]

    supported = sum(
        x["verdict"] == "SUPPORTED"
        for x in subset
    )

    contradicted = sum(
        x["verdict"] == "CONTRADICTED"
        for x in subset
    )

    insufficient = sum(
        x["verdict"]
        == "INSUFFICIENT_EVIDENCE"
        for x in subset
    )

    total = len(
        subset
    )

    return {
        "checkpoint": checkpoint,
        "total_claims": total,

        "supported": supported,
        "contradicted": contradicted,
        "insufficient_evidence": (
            insufficient
        ),

        "support_rate": (
            supported / total
            if total
            else None
        ),

        "contradiction_rate": (
            contradicted / total
            if total
            else None
        ),

        "insufficient_evidence_rate": (
            insufficient / total
            if total
            else None
        ),

        "mean_max_entailment": (
            float(
                np.mean(
                    [
                        x["max_entailment"]
                        for x in subset
                    ]
                )
            )
            if subset
            else None
        ),

        "mean_max_contradiction": (
            float(
                np.mean(
                    [
                        x[
                            "max_contradiction"
                        ]
                        for x in subset
                    ]
                )
            )
            if subset
            else None
        ),

        "mean_top2_entailment": (
            float(
                np.mean(
                    [
                        x[
                            "top2_entailment_mean"
                        ]
                        for x in subset
                    ]
                )
            )
            if subset
            else None
        ),

        "mean_top2_contradiction": (
            float(
                np.mean(
                    [
                        x[
                            "top2_contradiction_mean"
                        ]
                        for x in subset
                    ]
                )
            )
            if subset
            else None
        ),
    }


def write_summary(
    results,
):

    checkpoints = sorted(
        {
            x["checkpoint"]
            for x in results
        }
    )

    summaries = [
        checkpoint_summary(
            checkpoint,
            results,
        )
        for checkpoint
        in checkpoints
    ]

    payload = {
        "model": MODEL_NAME,

        "retrieval_version": "v2",

        "settings": {
            "top_k_evidence": (
                TOP_K_EVIDENCE
            ),
            "hard_remedy_filter": True,
            "max_length": (
                MAX_LENGTH
            ),
            "min_decisive_probability": (
                MIN_DECISIVE_PROBABILITY
            ),
            "min_decisive_margin": (
                MIN_DECISIVE_MARGIN
            ),
        },

        "results": summaries,
    }

    save_json(
        SUMMARY_JSON_FILE,
        payload,
    )

    with SUMMARY_CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summaries[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )

    return summaries


def print_summary(
    summaries,
):

    print()
    print("=" * 90)
    print(
        "NLI FACTUALITY V2 SUMMARY"
    )
    print("=" * 90)

    for item in summaries:

        print()
        print(
            item["checkpoint"]
        )

        print(
            f"  Claims:                  "
            f"{item['total_claims']}"
        )

        print(
            f"  Supported:               "
            f"{item['supported']}"
        )

        print(
            f"  Contradicted:            "
            f"{item['contradicted']}"
        )

        print(
            f"  Insufficient evidence:   "
            f"{item['insufficient_evidence']}"
        )

        print(
            f"  Support rate:            "
            f"{item['support_rate']:.3f}"
        )

        print(
            f"  Contradiction rate:      "
            f"{item['contradiction_rate']:.3f}"
        )

        print(
            f"  Insufficient rate:       "
            f"{item['insufficient_evidence_rate']:.3f}"
        )

        print(
            f"  Mean max entailment:     "
            f"{item['mean_max_entailment']:.3f}"
        )

        print(
            f"  Mean max contradiction:  "
            f"{item['mean_max_contradiction']:.3f}"
        )

        print(
            f"  Mean top-2 entailment:   "
            f"{item['mean_top2_entailment']:.3f}"
        )

        print(
            f"  Mean top-2 contradiction:"
            f" {item['mean_top2_contradiction']:.3f}"
        )

    print()
    print(
        f"Detailed results:\n"
        f"{DETAILED_RESULTS_FILE}"
    )

    print(
        f"\nSummary:\n"
        f"{SUMMARY_JSON_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print(
        "EVALUATION 4A V2 — "
        "REMEDY-AWARE RETRIEVAL + NLI"
    )
    print("=" * 90)
    print()

    (
        claims,
        passages,
        passage_embeddings,
        claim_embeddings,
    ) = load_data()

    (
        tokenizer,
        model,
        device,
    ) = load_nli_model()

    print(
        f"NLI comparisons: "
        f"{len(claims) * TOP_K_EVIDENCE:,}"
    )
    print()

    results = evaluate_all(
        claims=claims,
        passages=passages,
        passage_embeddings=(
            passage_embeddings
        ),
        claim_embeddings=(
            claim_embeddings
        ),
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    summaries = write_summary(
        results
    )

    print_summary(
        summaries
    )


if __name__ == "__main__":
    main()
