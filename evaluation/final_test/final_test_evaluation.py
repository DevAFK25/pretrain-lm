from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model.language_model import SmallLanguageModel


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BASE_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "base"
    / "best.pt"
)

SELECTED_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "domain_v1"
    / "epoch_4.pt"
)

DOMAIN_TEST = (
    PROJECT_ROOT
    / "data"
    / "domain"
    / "domain_test.pt"
)

WIKITEXT_TEST = (
    PROJECT_ROOT
    / "data"
    / "wikitext_test.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "final_test_results.json"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "final_test_results.csv"
)

EVAL_BATCH_SIZE = 32


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# MODEL
# ============================================================

ARCH_KEYS = [
    "vocab_size",
    "block_size",
    "embedding_dim",
    "num_heads",
    "num_layers",
    "feed_forward_dim",
]


def load_checkpoint(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def create_model_from_checkpoint(
    checkpoint,
    device,
):
    config = checkpoint["config"]

    model = SmallLanguageModel(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        feed_forward_dim=config["feed_forward_dim"],
        num_layers=config["num_layers"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


# ============================================================
# TEST DATA
# ============================================================

def make_test_loader(
    path,
    block_size,
):
    tokens = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(tokens, torch.Tensor):
        raise TypeError(
            f"{path} is not a tensor."
        )

    tokens = tokens.long().flatten()

    # Each example needs:
    #
    # token 0 ... token 255 -> input
    # token 1 ... token 256 -> target
    #
    # So each raw window has block_size + 1 tokens.
    #
    # Windows advance by block_size tokens.
    num_windows = (
        (tokens.numel() - 1)
        // block_size
    )

    usable_tokens = (
        num_windows * block_size + 1
    )

    tokens = tokens[
        :usable_tokens
    ]

    windows = tokens.unfold(
        dimension=0,
        size=block_size + 1,
        step=block_size,
    )

    inputs = windows[:, :-1]
    targets = windows[:, 1:]

    dataset = TensorDataset(
        inputs,
        targets,
    )

    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    metadata = {
        "raw_tokens": int(
            usable_tokens
        ),
        "evaluation_windows": int(
            num_windows
        ),
        "predicted_tokens": int(
            num_windows * block_size
        ),
    }

    return loader, metadata


# ============================================================
# LOSS
# ============================================================

@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
    vocab_size,
):
    total_loss = 0.0
    total_tokens = 0

    for inputs, targets in loader:
        inputs = inputs.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        logits, _ = model(inputs)

        # Sum rather than mean so that the final result
        # is a true token-weighted cross-entropy.
        loss = F.cross_entropy(
            logits.reshape(
                -1,
                vocab_size,
            ),
            targets.reshape(-1),
            reduction="sum",
        )

        total_loss += loss.item()
        total_tokens += targets.numel()

    mean_loss = (
        total_loss
        / total_tokens
    )

    perplexity = math.exp(
        mean_loss
    )

    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "evaluated_tokens": total_tokens,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    required_files = [
        BASE_CHECKPOINT,
        SELECTED_CHECKPOINT,
        DOMAIN_TEST,
        WIKITEXT_TEST,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )
            
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print("=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)
    print()
    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()

    base_ckpt = load_checkpoint(
        BASE_CHECKPOINT
    )

    final_ckpt = load_checkpoint(
        SELECTED_CHECKPOINT
    )

    # --------------------------------------------------------
    # Verify architecture equality
    # --------------------------------------------------------

    for key in ARCH_KEYS:
        a = base_ckpt["config"][key]
        b = final_ckpt["config"][key]

        if a != b:
            raise RuntimeError(
                f"Architecture mismatch "
                f"for {key}: "
                f"base={a}, final={b}"
            )

    config = base_ckpt["config"]

    block_size = config[
        "block_size"
    ]

    vocab_size = config[
        "vocab_size"
    ]

    print(
        f"Block size:     {block_size}"
    )
    print(
        f"Vocabulary:     {vocab_size:,}"
    )

    # --------------------------------------------------------
    # Build test loaders
    # --------------------------------------------------------

    domain_loader, domain_meta = (
        make_test_loader(
            DOMAIN_TEST,
            block_size,
        )
    )

    wiki_loader, wiki_meta = (
        make_test_loader(
            WIKITEXT_TEST,
            block_size,
        )
    )

    print()
    print("DOMAIN TEST")
    print(
        f"  Windows:          "
        f"{domain_meta['evaluation_windows']:,}"
    )
    print(
        f"  Predicted tokens: "
        f"{domain_meta['predicted_tokens']:,}"
    )

    print()
    print("WIKITEXT TEST")
    print(
        f"  Windows:          "
        f"{wiki_meta['evaluation_windows']:,}"
    )
    print(
        f"  Predicted tokens: "
        f"{wiki_meta['predicted_tokens']:,}"
    )

    # --------------------------------------------------------
    # BASE MODEL
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("BASE MODEL")
    print("=" * 80)

    base_model = create_model_from_checkpoint(
        base_ckpt,
        device,
    )

    parameter_count = count_parameters(
        base_model
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    print("\nEvaluating domain test...")
    base_domain = evaluate(
        base_model,
        domain_loader,
        device,
        vocab_size,
    )

    print(
        f"  Loss:       "
        f"{base_domain['loss']:.4f}"
    )
    print(
        f"  Perplexity: "
        f"{base_domain['perplexity']:.2f}"
    )

    print("\nEvaluating WikiText test...")
    base_wiki = evaluate(
        base_model,
        wiki_loader,
        device,
        vocab_size,
    )

    print(
        f"  Loss:       "
        f"{base_wiki['loss']:.4f}"
    )
    print(
        f"  Perplexity: "
        f"{base_wiki['perplexity']:.2f}"
    )

    del base_model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # FINAL E4 MODEL
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL DOMAIN MODEL — EPOCH 4")
    print("=" * 80)

    final_model = create_model_from_checkpoint(
        final_ckpt,
        device,
    )

    print("\nEvaluating domain test...")
    final_domain = evaluate(
        final_model,
        domain_loader,
        device,
        vocab_size,
    )

    print(
        f"  Loss:       "
        f"{final_domain['loss']:.4f}"
    )
    print(
        f"  Perplexity: "
        f"{final_domain['perplexity']:.2f}"
    )

    print("\nEvaluating WikiText test...")
    final_wiki = evaluate(
        final_model,
        wiki_loader,
        device,
        vocab_size,
    )

    print(
        f"  Loss:       "
        f"{final_wiki['loss']:.4f}"
    )
    print(
        f"  Perplexity: "
        f"{final_wiki['perplexity']:.2f}"
    )

    # --------------------------------------------------------
    # COMPARISONS
    # --------------------------------------------------------

    domain_loss_reduction = (
        (
            base_domain["loss"]
            - final_domain["loss"]
        )
        / base_domain["loss"]
        * 100
    )

    domain_ppl_reduction = (
        (
            base_domain["perplexity"]
            - final_domain["perplexity"]
        )
        / base_domain["perplexity"]
        * 100
    )

    wiki_loss_change = (
        final_wiki["loss"]
        - base_wiki["loss"]
    )

    wiki_ppl_change_pct = (
        (
            final_wiki["perplexity"]
            - base_wiki["perplexity"]
        )
        / base_wiki["perplexity"]
        * 100
    )

    print()
    print("=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print()
    print("DOMAIN ADAPTATION")
    print(
        f"  Base loss:              "
        f"{base_domain['loss']:.4f}"
    )
    print(
        f"  Final E4 loss:          "
        f"{final_domain['loss']:.4f}"
    )
    print(
        f"  Loss reduction:         "
        f"{domain_loss_reduction:.2f}%"
    )

    print()
    print(
        f"  Base perplexity:        "
        f"{base_domain['perplexity']:.2f}"
    )
    print(
        f"  Final E4 perplexity:    "
        f"{final_domain['perplexity']:.2f}"
    )
    print(
        f"  Perplexity reduction:   "
        f"{domain_ppl_reduction:.2f}%"
    )

    print()
    print("GENERAL-LANGUAGE RETENTION")
    print(
        f"  Base WikiText loss:     "
        f"{base_wiki['loss']:.4f}"
    )
    print(
        f"  Final WikiText loss:    "
        f"{final_wiki['loss']:.4f}"
    )
    print(
        f"  Loss change:            "
        f"{wiki_loss_change:+.4f}"
    )

    print()
    print(
        f"  Base WikiText PPL:      "
        f"{base_wiki['perplexity']:.2f}"
    )
    print(
        f"  Final WikiText PPL:     "
        f"{final_wiki['perplexity']:.2f}"
    )
    print(
        f"  Perplexity change:      "
        f"{wiki_ppl_change_pct:+.2f}%"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    payload = {
        "selected_checkpoint": str(
            SELECTED_CHECKPOINT
        ),
        "base_checkpoint": str(
            BASE_CHECKPOINT
        ),

        "architecture": {
            key: config[key]
            for key in ARCH_KEYS
        },

        "parameter_count": (
            parameter_count
        ),

        "datasets": {
            "domain_test": domain_meta,
            "wikitext_test": wiki_meta,
        },

        "base": {
            "domain_test": base_domain,
            "wikitext_test": base_wiki,
        },

        "final_epoch_4": {
            "domain_test": final_domain,
            "wikitext_test": final_wiki,
        },

        "comparison": {
            "domain_loss_reduction_percent":
                domain_loss_reduction,

            "domain_perplexity_reduction_percent":
                domain_ppl_reduction,

            "wikitext_loss_change":
                wiki_loss_change,

            "wikitext_perplexity_change_percent":
                wiki_ppl_change_pct,
        },
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = [
        {
            "model": "base",
            "dataset": "domain_test",
            **base_domain,
        },
        {
            "model": "base",
            "dataset": "wikitext_test",
            **base_wiki,
        },
        {
            "model": "epoch_4",
            "dataset": "domain_test",
            **final_domain,
        },
        {
            "model": "epoch_4",
            "dataset": "wikitext_test",
            **final_wiki,
        },
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "dataset",
                "loss",
                "perplexity",
                "evaluated_tokens",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
