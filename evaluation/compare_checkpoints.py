from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from model.language_model import SmallLanguageModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_FILE = (
    PROJECT_ROOT / "tokenizer" / "output" / "tokenizer.json"
)

CHECKPOINT_DIRECTORY = (
    PROJECT_ROOT / "checkpoints" / "domain_v1"
)

CHECKPOINTS = [
    "epoch_3.pt",
    "epoch_4.pt",
    "epoch_5.pt",
]

# Deliberately mixed:
# - structured domain continuations
# - free domain prose
# - general-language controls
PROMPTS = [
    "<REMEDY> Bryonia",
    "<REMEDY> Nux vomica",
    "<REMEDY> Pulsatilla",
    "<REMEDY> Sulphur",
    "The symptoms are",
    "The patient complains of",
    "Headache with",
    "The remedy is indicated when",
    "The history of medicine",
    "The weather today is",
]

GENERATION_TOKENS = 120
TEMPERATURE = 0.8
TOP_K = 40
SEED = 42


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model(config, device):
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
    model,
    tokenizer,
    prompt,
    block_size,
    device,
):
    # Reset seed for EVERY prompt/checkpoint.
    # This makes the comparison much fairer.
    random.seed(SEED)
    torch.manual_seed(SEED)

    if device.type == "mps":
        torch.mps.manual_seed(SEED)

    prompt_ids = tokenizer.encode(prompt).ids

    if not prompt_ids:
        raise ValueError(
            f"Prompt produced no tokens: {prompt!r}"
        )

    generated_ids = list(prompt_ids)

    model.eval()

    for _ in range(GENERATION_TOKENS):
        context = generated_ids[-block_size:]

        input_ids = torch.tensor(
            [context],
            dtype=torch.long,
            device=device,
        )

        logits, _ = model(input_ids)

        next_logits = (
            logits[:, -1, :] / TEMPERATURE
        )

        top_values, top_indices = torch.topk(
            next_logits,
            k=min(TOP_K, next_logits.shape[-1]),
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
            -1,
            sampled_position,
        )

        token_id = int(next_token.item())

        if token_id == 0:
            break

        generated_ids.append(token_id)

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
    )


def main():
    device = get_device()
    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )

    print(f"Device: {device}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Top-k: {TOP_K}")
    print(f"Seed: {SEED}")
    print()

    output_directory = (
        PROJECT_ROOT / "evaluation" / "outputs"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for checkpoint_name in CHECKPOINTS:
        checkpoint_path = (
            CHECKPOINT_DIRECTORY / checkpoint_name
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

        header = (
            f"{'=' * 80}\n"
            f"CHECKPOINT: {checkpoint_name}\n"
            f"Domain epoch: "
            f"{checkpoint.get('domain_epoch')}\n"
            f"Domain validation loss: "
            f"{checkpoint.get('domain_validation_loss')}\n"
            f"WikiText validation loss: "
            f"{checkpoint.get('wikitext_validation_loss')}\n"
            f"{'=' * 80}\n"
        )

        print(header)

        output_parts = [header]

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

            section = (
                f"\nPROMPT {index}: {prompt!r}\n"
                f"{'-' * 80}\n"
                f"{generated}\n"
            )

            print(section)
            output_parts.append(section)

        output_file = (
            output_directory
            / f"{checkpoint_path.stem}_generations.txt"
        )

        output_file.write_text(
            "\n".join(output_parts),
            encoding="utf-8",
        )

        del model

        if device.type == "mps":
            torch.mps.empty_cache()

    print()
    print(
        "Comparison complete. Outputs saved to:"
    )
    print(output_directory)


if __name__ == "__main__":
    main()