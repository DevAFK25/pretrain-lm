from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model.language_model import SmallLanguageModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TOKENIZER_FILE = (
    PROJECT_ROOT
    / "tokenizer"
    / "output"
    / "tokenizer.json"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "domain_v1"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "structure"
    / "outputs"
)

CHECKPOINTS = [
    "epoch_3.pt",
    "epoch_4.pt",
    "epoch_5.pt",
]

PROMPTS = [
    "<REMEDY> Bryonia",
    "<REMEDY> Nux vomica",
    "<REMEDY> Pulsatilla",
    "<REMEDY> Sulphur",
    "<REMEDY> Aconitum",
    "<REMEDY> Belladonna",
    "<SECTION> Mind",
    "<SECTION> Head",
    "<SECTION> Eyes",
    "<SECTION> Stomach",
    "<SECTION> Respiratory Organs",
    "<SECTION> Generalities",
]

GENERATION_TOKENS = 120
TEMPERATURE = 0.8
TOP_K = 40
SEED = 42

# Section names observed / expected from our corpus style.
KNOWN_SECTION_NAMES = {
    "Mind",
    "Head",
    "Eyes",
    "Ears",
    "Nose",
    "Face",
    "Mouth",
    "Throat",
    "Appetite",
    "Stomach",
    "Abdomen",
    "Stool",
    "Stool and Anus",
    "Urinary Organs",
    "Male Sexual Organs",
    "Female Sexual Organs",
    "Respiratory Organs",
    "Chest",
    "Heart",
    "Heart and Pulse",
    "Neck and Back",
    "Back",
    "Upper Limbs",
    "Lower Limbs",
    "Generalities",
    "Skin",
    "Sleep",
    "Fever",
    "Clinical",
    "Characteristics",
    "Relations",
    "Causation",
    "Natural History",
    "Text",
}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_model(
    config: dict,
    device: torch.device,
) -> SmallLanguageModel:

    model = SmallLanguageModel(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        feed_forward_dim=config["feed_forward_dim"],
        num_layers=config["num_layers"],
    )

    return model.to(device)


@torch.no_grad()
def generate(
    model: SmallLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    block_size: int,
    device: torch.device,
) -> str:

    torch.manual_seed(SEED)

    if device.type == "mps":
        torch.mps.manual_seed(SEED)

    prompt_ids = tokenizer.encode(prompt).ids

    generated_ids = list(prompt_ids)

    model.eval()

    for _ in range(GENERATION_TOKENS):

        context_ids = generated_ids[
            -block_size:
        ]

        input_ids = torch.tensor(
            [context_ids],
            dtype=torch.long,
            device=device,
        )

        logits, _ = model(input_ids)

        next_logits = (
            logits[:, -1, :]
            / TEMPERATURE
        )

        top_values, top_indices = torch.topk(
            next_logits,
            k=min(
                TOP_K,
                next_logits.shape[-1],
            ),
        )

        probabilities = torch.softmax(
            top_values,
            dim=-1,
        )

        sampled_position = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        next_token = top_indices.gather(
            dim=-1,
            index=sampled_position,
        )

        next_token_id = int(
            next_token.item()
        )

        if next_token_id == 0:
            break

        generated_ids.append(
            next_token_id
        )

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
    )


def extract_section_names(
    text: str,
) -> list[str]:

    return [
        match.strip()
        for match in re.findall(
            r"<SECTION>\s*([^\n<>]+)",
            text,
        )
    ]


def analyse_generation(
    text: str,
    prompt: str,
) -> dict:

    # Remove the prompt itself.
    if text.startswith(prompt):
        continuation = text[len(prompt):]
    else:
        continuation = text

    # Only markers GENERATED after the prompt count.
    section_names = [
        match.strip()
        for match in re.findall(
            r"<SECTION>\s*([^\n<>]+)",
            continuation,
        )
    ]

    remedy_names = [
        match.strip()
        for match in re.findall(
            r"<REMEDY>\s*([^\n<>]+)",
            continuation,
        )
    ]

    valid_sections = [
        name
        for name in section_names
        if name in KNOWN_SECTION_NAMES
    ]

    unknown_sections = [
        name
        for name in section_names
        if name not in KNOWN_SECTION_NAMES
    ]

    malformed_section_markers = len(
        re.findall(
            r"<SECTION>(?!\s*[^\n<>]+)",
            continuation,
        )
    )

    malformed_remedy_markers = len(
        re.findall(
            r"<REMEDY>(?!\s*[^\n<>]+)",
            continuation,
        )
    )

    consecutive_section_repeats = sum(
        previous == current
        for previous, current in zip(
            section_names,
            section_names[1:],
        )
    )

    return {
        "prompt": prompt,

        # Important: these are GENERATED markers only.
        "generated_section_markers": len(section_names),
        "generated_remedy_markers": len(remedy_names),

        "valid_generated_sections": len(valid_sections),
        "unknown_generated_sections": len(unknown_sections),

        "valid_sections": valid_sections,
        "unknown_sections": unknown_sections,

        "unique_valid_sections": sorted(
            set(valid_sections)
        ),

        "malformed_section_markers": malformed_section_markers,
        "malformed_remedy_markers": malformed_remedy_markers,

        "consecutive_section_repeats": consecutive_section_repeats,

        # Useful specifically for <REMEDY> prompts.
        "remedy_to_section_transition": (
            prompt.startswith("<REMEDY>")
            and len(valid_sections) > 0
        ),
    }


def summarize_checkpoint(
    analyses: list[dict],
) -> dict:

    generated_sections = sum(
        item["generated_section_markers"]
        for item in analyses
    )

    valid_sections = sum(
        item["valid_generated_sections"]
        for item in analyses
    )

    unknown_sections = sum(
        item["unknown_generated_sections"]
        for item in analyses
    )

    remedy_prompts = [
        item
        for item in analyses
        if item["prompt"].startswith("<REMEDY>")
    ]

    remedy_transitions = sum(
        item["remedy_to_section_transition"]
        for item in remedy_prompts
    )

    unique_valid_sections = sorted(
        {
            section
            for item in analyses
            for section in item["unique_valid_sections"]
        }
    )

    return {
        "total_prompts": len(analyses),

        "generated_section_markers": generated_sections,
        "generated_remedy_markers": sum(
            item["generated_remedy_markers"]
            for item in analyses
        ),

        "valid_generated_sections": valid_sections,
        "unknown_generated_sections": unknown_sections,

        "generated_section_validity_rate": (
            valid_sections / generated_sections
            if generated_sections > 0
            else None
        ),

        "unique_valid_generated_sections": unique_valid_sections,
        "unique_valid_generated_section_count": len(
            unique_valid_sections
        ),

        "remedy_prompts": len(remedy_prompts),
        "remedy_to_section_transitions": remedy_transitions,

        "remedy_to_section_transition_rate": (
            remedy_transitions / len(remedy_prompts)
            if remedy_prompts
            else None
        ),

        "malformed_section_markers": sum(
            item["malformed_section_markers"]
            for item in analyses
        ),

        "malformed_remedy_markers": sum(
            item["malformed_remedy_markers"]
            for item in analyses
        ),

        "consecutive_section_repeats": sum(
            item["consecutive_section_repeats"]
            for item in analyses
        ),
    }


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )

    device = get_device()

    print(
        f"Using device: {device}"
    )

    overall_results = {}

    for checkpoint_name in CHECKPOINTS:

        checkpoint_path = (
            CHECKPOINT_DIR
            / checkpoint_name
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        config = checkpoint["config"]

        model = create_model(
            config=config,
            device=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print()
        print("=" * 80)
        print(
            f"CHECKPOINT: {checkpoint_name}"
        )
        print("=" * 80)

        checkpoint_analyses = []
        raw_sections = []

        for index, prompt in enumerate(
            PROMPTS,
            start=1,
        ):

            generated = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                block_size=config["block_size"],
                device=device,
            )

            analysis = analyse_generation(
                text=generated,
                prompt=prompt,
            )

            checkpoint_analyses.append(
                analysis
            )

            raw_sections.append(
                "\n".join(
                    [
                        f"PROMPT {index}: {prompt!r}",
                        "-" * 80,
                        generated,
                        "",
                        "STRUCTURE ANALYSIS:",
                        json.dumps(
                            analysis,
                            indent=2,
                            ensure_ascii=False,
                        ),
                        "",
                    ]
                )
            )

            print(
                f"{index:02d}. "
                f"{prompt:<32} | "
                f"new_sections={analysis['generated_section_markers']:2d} | "
                f"valid={analysis['valid_generated_sections']:2d} | "
                f"unknown={analysis['unknown_generated_sections']:2d} | "
                f"transition={str(analysis['remedy_to_section_transition']):5}"
            )

        summary = summarize_checkpoint(
            checkpoint_analyses
        )

        overall_results[
            checkpoint_name
        ] = {
            "domain_validation_loss": (
                checkpoint.get(
                    "domain_validation_loss"
                )
            ),
            "wikitext_validation_loss": (
                checkpoint.get(
                    "wikitext_validation_loss"
                )
            ),
            "summary": summary,
            "per_prompt": (
                checkpoint_analyses
            ),
        }

        text_output = (
            OUTPUT_DIR
            / f"{checkpoint_path.stem}_structure.txt"
        )

        text_output.write_text(
            "\n".join(raw_sections),
            encoding="utf-8",
        )

        print()
        print("SUMMARY:")
        print(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            )
        )

        del model

        if device.type == "mps":
            torch.mps.empty_cache()

    json_output = (
        OUTPUT_DIR
        / "structure_results.json"
    )

    json_output.write_text(
        json.dumps(
            overall_results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("STRUCTURE EVALUATION COMPLETE")
    print("=" * 80)
    print(
        f"Outputs saved to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()