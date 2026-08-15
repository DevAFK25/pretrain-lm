from dataclasses import asdict, dataclass
from pathlib import Path
import time

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from model.language_model import SmallLanguageModel
from training.prepare_dataset import NextTokenDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_TOKEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "domain"
    / "domain_train.pt"
)

DOMAIN_VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "domain"
    / "domain_validation.pt"
)

WIKITEXT_VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "wikitext_validation.pt"
)

TOKENIZER_FILE = (
    PROJECT_ROOT
    / "tokenizer"
    / "output"
    / "tokenizer.json"
)

BASE_CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "checkpoints"
    / "base"
    / "best.pt"
)

CHECKPOINT_DIRECTORY = (
    PROJECT_ROOT
    / "checkpoints"
    / "domain_v1"
)

LATEST_CHECKPOINT_FILE = (
    CHECKPOINT_DIRECTORY
    / "latest.pt"
)

BEST_CHECKPOINT_FILE = (
    CHECKPOINT_DIRECTORY
    / "best.pt"
)


@dataclass(frozen=True)
class TrainingConfig:
    # Architecture — must match WikiText V2 exactly.
    vocab_size: int = 24_000
    block_size: int = 256
    embedding_dim: int = 384
    num_heads: int = 6
    num_layers: int = 8
    feed_forward_dim: int = 1_536

    # Continued-pretraining optimisation.
    batch_size: int = 32
    learning_rate: float = 1e-4
    minimum_learning_rate: float = 1e-5
    weight_decay: float = 0.01

    # Initial ceiling. We will judge the curve before deciding
    # whether all 10 epochs are useful. setting 5 for now
    num_epochs: int = 5

    # Use 200 for a short integration test on Jarvis.
    # Use None for the real run.
    max_train_steps_per_epoch: int | None = None

    checkpoint_every_steps: int = 100
    log_every: int = 10
    random_seed: int = 42

    generation_prompt: str = "<REMEDY> Bryonia"
    generation_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 40

    end_of_text_token_id: int = 0


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_token_ids(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(
            f"Token file not found:\n{path}"
        )

    token_ids = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(token_ids, torch.Tensor):
        raise TypeError(
            f"{path.name} did not contain a torch.Tensor."
        )

    if token_ids.ndim != 1:
        raise ValueError(
            f"{path.name} must be one-dimensional. "
            f"Received {tuple(token_ids.shape)}."
        )

    if token_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError(
            f"{path.name} has unsupported dtype "
            f"{token_ids.dtype}."
        )

    return token_ids


def create_data_loader(
    token_ids: torch.Tensor,
    config: TrainingConfig,
    shuffle: bool,
    drop_last: bool,
    epoch_number: int = 0,
) -> DataLoader:
    dataset = NextTokenDataset(
        token_ids=token_ids,
        block_size=config.block_size,
    )

    generator = None

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(
            config.random_seed + epoch_number
        )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        generator=generator,
    )


def get_steps_per_epoch(
    data_loader: DataLoader,
    config: TrainingConfig,
) -> int:
    if config.max_train_steps_per_epoch is None:
        return len(data_loader)

    return min(
        len(data_loader),
        config.max_train_steps_per_epoch,
    )


def create_model(
    config: TrainingConfig,
    device: torch.device,
) -> SmallLanguageModel:
    model = SmallLanguageModel(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        feed_forward_dim=config.feed_forward_dim,
        num_layers=config.num_layers,
    )

    return model.to(device)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def calculate_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    vocab_size: int,
    loss_function: torch.nn.Module,
) -> torch.Tensor:
    return loss_function(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
    )


def validate_base_config(
    checkpoint: dict,
    config: TrainingConfig,
) -> None:
    base_config = checkpoint.get("config", {})

    keys = [
        "vocab_size",
        "block_size",
        "embedding_dim",
        "num_heads",
        "num_layers",
        "feed_forward_dim",
    ]

    for key in keys:
        expected = getattr(config, key)
        received = base_config.get(key)

        if received != expected:
            raise RuntimeError(
                f"Architecture mismatch for {key}: "
                f"base={received}, domain={expected}"
            )


def load_base_model(
    model: SmallLanguageModel,
    config: TrainingConfig,
    device: torch.device,
) -> dict:
    if not BASE_CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Base checkpoint not found:\n"
            f"{BASE_CHECKPOINT_FILE}"
        )

    checkpoint = torch.load(
        BASE_CHECKPOINT_FILE,
        map_location=device,
        weights_only=False,
    )

    validate_base_config(
        checkpoint=checkpoint,
        config=config,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print()
    print("Loaded WikiText base checkpoint:")
    print(
        f"  Base epoch:           "
        f"{checkpoint.get('epoch')}"
    )
    print(
        f"  Base global step:     "
        f"{checkpoint.get('global_step'):,}"
    )
    print(
        f"  Base validation loss: "
        f"{checkpoint.get('validation_loss'):.4f}"
    )

    return checkpoint


def save_checkpoint(
    checkpoint_path: Path,
    model: SmallLanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainingConfig,
    domain_epoch: int,
    step_in_epoch: int,
    domain_global_step: int,
    epoch_complete: bool,
    epoch_loss_sum: float,
    epoch_steps_completed: int,
    domain_validation_loss: float | None,
    wikitext_validation_loss: float | None,
    best_domain_validation_loss: float,
    base_checkpoint: dict,
) -> None:
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "domain_epoch": domain_epoch,
        "step_in_epoch": step_in_epoch,
        "domain_global_step": domain_global_step,
        "epoch_complete": epoch_complete,
        "epoch_loss_sum": epoch_loss_sum,
        "epoch_steps_completed": epoch_steps_completed,
        "domain_validation_loss": domain_validation_loss,
        "wikitext_validation_loss": wikitext_validation_loss,
        "best_domain_validation_loss": (
            best_domain_validation_loss
        ),
        "base_checkpoint": {
            "epoch": base_checkpoint.get("epoch"),
            "global_step": base_checkpoint.get(
                "global_step"
            ),
            "validation_loss": base_checkpoint.get(
                "validation_loss"
            ),
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": asdict(config),
    }

    temporary_path = checkpoint_path.with_suffix(
        ".tmp"
    )

    torch.save(
        checkpoint,
        temporary_path,
    )

    temporary_path.replace(
        checkpoint_path
    )


def load_domain_checkpoint(
    model: SmallLanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> dict | None:
    if not LATEST_CHECKPOINT_FILE.exists():
        return None

    print()
    print(
        "Resuming domain training from:"
        f"\n{LATEST_CHECKPOINT_FILE}"
    )

    checkpoint = torch.load(
        LATEST_CHECKPOINT_FILE,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    print(
        f"Domain checkpoint loaded at step "
        f"{checkpoint['domain_global_step']:,}."
    )

    return checkpoint


@torch.no_grad()
def evaluate(
    model: SmallLanguageModel,
    data_loader: DataLoader,
    loss_function: torch.nn.Module,
    config: TrainingConfig,
    device: torch.device,
) -> float:
    model.eval()

    total_loss = 0.0
    completed_batches = 0

    for input_ids, target_ids in data_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits, _ = model(input_ids)

        loss = calculate_loss(
            logits=logits,
            target_ids=target_ids,
            vocab_size=config.vocab_size,
            loss_function=loss_function,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Validation loss became non-finite."
            )

        total_loss += loss.item()
        completed_batches += 1

    if completed_batches == 0:
        raise RuntimeError(
            "Validation produced no batches."
        )

    return total_loss / completed_batches


@torch.no_grad()
def generate_text(
    model: SmallLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    config: TrainingConfig,
    device: torch.device,
) -> str:
    model.eval()

    prompt_ids = tokenizer.encode(
        prompt
    ).ids

    if not prompt_ids:
        raise ValueError(
            "Generation prompt produced no tokens."
        )

    generated_ids = list(prompt_ids)

    for _ in range(
        config.generation_tokens
    ):
        context_ids = generated_ids[
            -config.block_size:
        ]

        input_ids = torch.tensor(
            [context_ids],
            dtype=torch.long,
            device=device,
        )

        logits, _ = model(input_ids)

        next_token_logits = (
            logits[:, -1, :]
            / config.temperature
        )

        if config.top_k > 0:
            top_values, top_indices = torch.topk(
                next_token_logits,
                k=min(
                    config.top_k,
                    next_token_logits.shape[-1],
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

        else:
            probabilities = torch.softmax(
                next_token_logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

        next_token_id = int(
            next_token.item()
        )

        if (
            next_token_id
            == config.end_of_text_token_id
        ):
            break

        generated_ids.append(
            next_token_id
        )

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
    )


def train_one_epoch(
    model: SmallLanguageModel,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_function: torch.nn.Module,
    config: TrainingConfig,
    device: torch.device,
    epoch_number: int,
    domain_global_step: int,
    resume_step: int,
    epoch_loss_sum: float,
    epoch_steps_completed: int,
    best_domain_validation_loss: float,
    base_checkpoint: dict,
) -> tuple[float, int, float, int]:
    model.train()

    steps_per_epoch = get_steps_per_epoch(
        data_loader=data_loader,
        config=config,
    )

    total_tokens = 0

    synchronize_device(device)
    start_time = time.perf_counter()

    for batch_number, (
        input_ids,
        target_ids,
    ) in enumerate(
        data_loader,
        start=1,
    ):
        if batch_number <= resume_step:
            continue

        if batch_number > steps_per_epoch:
            break

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits, _ = model(input_ids)

        loss = calculate_loss(
            logits=logits,
            target_ids=target_ids,
            vocab_size=config.vocab_size,
            loss_function=loss_function,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at domain "
                f"epoch {epoch_number}, "
                f"step {batch_number}."
            )

        loss.backward()
        optimizer.step()
        scheduler.step()

        domain_global_step += 1
        epoch_steps_completed += 1
        epoch_loss_sum += loss.item()
        total_tokens += input_ids.numel()

        if (
            batch_number == 1
            or batch_number
            % config.log_every == 0
            or batch_number == steps_per_epoch
        ):
            synchronize_device(device)

            elapsed = (
                time.perf_counter()
                - start_time
            )

            average_loss = (
                epoch_loss_sum
                / epoch_steps_completed
            )

            learning_rate = (
                optimizer.param_groups[0]["lr"]
            )

            tokens_per_second = (
                total_tokens / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"Domain epoch {epoch_number} | "
                f"Step {batch_number:4d}/"
                f"{steps_per_epoch:4d} | "
                f"Loss {loss.item():.4f} | "
                f"Average {average_loss:.4f} | "
                f"LR {learning_rate:.7f} | "
                f"{tokens_per_second:,.0f} "
                f"tokens/sec"
            )

        if (
            domain_global_step
            % config.checkpoint_every_steps
            == 0
        ):
            save_checkpoint(
                checkpoint_path=(
                    LATEST_CHECKPOINT_FILE
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                domain_epoch=epoch_number,
                step_in_epoch=batch_number,
                domain_global_step=(
                    domain_global_step
                ),
                epoch_complete=False,
                epoch_loss_sum=epoch_loss_sum,
                epoch_steps_completed=(
                    epoch_steps_completed
                ),
                domain_validation_loss=None,
                wikitext_validation_loss=None,
                best_domain_validation_loss=(
                    best_domain_validation_loss
                ),
                base_checkpoint=base_checkpoint,
            )

            print(
                f"Progress checkpoint saved at "
                f"domain step "
                f"{domain_global_step:,}."
            )

    if epoch_steps_completed == 0:
        raise RuntimeError(
            "No training steps were completed."
        )

    average_loss = (
        epoch_loss_sum
        / epoch_steps_completed
    )

    return (
        average_loss,
        domain_global_step,
        epoch_loss_sum,
        epoch_steps_completed,
    )


def main() -> None:
    config = TrainingConfig()

    torch.manual_seed(
        config.random_seed
    )

    device = get_device()

    print(
        f"Using device:                "
        f"{device}"
    )
    print(
        f"Domain epochs:               "
        f"{config.num_epochs}"
    )
    print(
        f"Maximum steps/epoch:         "
        f"{config.max_train_steps_per_epoch}"
    )

    train_token_ids = load_token_ids(
        TRAIN_TOKEN_FILE
    )

    domain_validation_ids = load_token_ids(
        DOMAIN_VALIDATION_FILE
    )

    wikitext_validation_ids = load_token_ids(
        WIKITEXT_VALIDATION_FILE
    )

    train_loader_first = create_data_loader(
        token_ids=train_token_ids,
        config=config,
        shuffle=True,
        drop_last=True,
        epoch_number=1,
    )

    steps_per_epoch = get_steps_per_epoch(
        train_loader_first,
        config,
    )

    total_training_steps = (
        steps_per_epoch
        * config.num_epochs
    )

    domain_validation_loader = create_data_loader(
        token_ids=domain_validation_ids,
        config=config,
        shuffle=False,
        drop_last=False,
    )

    wikitext_validation_loader = create_data_loader(
        token_ids=wikitext_validation_ids,
        config=config,
        shuffle=False,
        drop_last=False,
    )

    model = create_model(
        config=config,
        device=device,
    )

    base_checkpoint = load_base_model(
        model=model,
        config=config,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_training_steps,
            eta_min=config.minimum_learning_rate,
        )
    )

    loss_function = (
        torch.nn.CrossEntropyLoss()
    )

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )

    print()
    print(
        f"Domain train tokens:         "
        f"{train_token_ids.numel():,}"
    )
    print(
        f"Domain validation tokens:    "
        f"{domain_validation_ids.numel():,}"
    )
    print(
        f"WikiText validation tokens:  "
        f"{wikitext_validation_ids.numel():,}"
    )
    print(
        f"Steps per domain epoch:      "
        f"{steps_per_epoch:,}"
    )
    print(
        f"Maximum domain steps:        "
        f"{total_training_steps:,}"
    )
    print(
        f"Model parameters:            "
        f"{count_parameters(model):,}"
    )

    domain_checkpoint = (
        load_domain_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
    )

    start_epoch = 1
    resume_step = 0
    domain_global_step = 0
    epoch_loss_sum = 0.0
    epoch_steps_completed = 0
    best_domain_validation_loss = (
        float("inf")
    )

    if domain_checkpoint is None:
        print()
        print(
            "===== PRE-ADAPTATION BASELINE ====="
        )

        baseline_domain_loss = evaluate(
            model=model,
            data_loader=(
                domain_validation_loader
            ),
            loss_function=loss_function,
            config=config,
            device=device,
        )

        baseline_wikitext_loss = evaluate(
            model=model,
            data_loader=(
                wikitext_validation_loader
            ),
            loss_function=loss_function,
            config=config,
            device=device,
        )

        print(
            f"Domain validation loss:      "
            f"{baseline_domain_loss:.4f}"
        )
        print(
            f"WikiText validation loss:    "
            f"{baseline_wikitext_loss:.4f}"
        )

        print()
        print(
            "Starting domain adaptation "
            "with a fresh optimizer."
        )

    else:
        domain_global_step = (
            domain_checkpoint[
                "domain_global_step"
            ]
        )

        best_domain_validation_loss = (
            domain_checkpoint.get(
                "best_domain_validation_loss",
                float("inf"),
            )
        )

        if domain_checkpoint[
            "epoch_complete"
        ]:
            start_epoch = (
                domain_checkpoint[
                    "domain_epoch"
                ]
                + 1
            )
        else:
            start_epoch = (
                domain_checkpoint[
                    "domain_epoch"
                ]
            )

            resume_step = (
                domain_checkpoint[
                    "step_in_epoch"
                ]
            )

            epoch_loss_sum = (
                domain_checkpoint.get(
                    "epoch_loss_sum",
                    0.0,
                )
            )

            epoch_steps_completed = (
                domain_checkpoint.get(
                    "epoch_steps_completed",
                    resume_step,
                )
            )

    if start_epoch > config.num_epochs:
        print(
            "\nDomain training is "
            "already complete."
        )
        return

    for epoch_number in range(
        start_epoch,
        config.num_epochs + 1,
    ):
        print()
        print("=" * 70)
        print(
            f"DOMAIN EPOCH "
            f"{epoch_number}/"
            f"{config.num_epochs}"
        )
        print("=" * 70)

        train_loader = create_data_loader(
            token_ids=train_token_ids,
            config=config,
            shuffle=True,
            drop_last=True,
            epoch_number=epoch_number,
        )

        (
            training_loss,
            domain_global_step,
            epoch_loss_sum,
            epoch_steps_completed,
        ) = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            config=config,
            device=device,
            epoch_number=epoch_number,
            domain_global_step=(
                domain_global_step
            ),
            resume_step=resume_step,
            epoch_loss_sum=epoch_loss_sum,
            epoch_steps_completed=(
                epoch_steps_completed
            ),
            best_domain_validation_loss=(
                best_domain_validation_loss
            ),
            base_checkpoint=base_checkpoint,
        )

        domain_validation_loss = evaluate(
            model=model,
            data_loader=(
                domain_validation_loader
            ),
            loss_function=loss_function,
            config=config,
            device=device,
        )

        wikitext_validation_loss = evaluate(
            model=model,
            data_loader=(
                wikitext_validation_loader
            ),
            loss_function=loss_function,
            config=config,
            device=device,
        )

        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=config.generation_prompt,
            config=config,
            device=device,
        )

        print()
        print(
            f"Training loss:               "
            f"{training_loss:.4f}"
        )
        print(
            f"Domain validation loss:      "
            f"{domain_validation_loss:.4f}"
        )
        print(
            f"WikiText validation loss:    "
            f"{wikitext_validation_loss:.4f}"
        )
        print(
            f"Domain global step:          "
            f"{domain_global_step:,}"
        )

        print()
        print("Generated sample:")
        print("-" * 70)
        print(generated_text)
        print("-" * 70)

        if (
            domain_validation_loss
            < best_domain_validation_loss
        ):
            best_domain_validation_loss = (
                domain_validation_loss
            )

            save_checkpoint(
                checkpoint_path=(
                    BEST_CHECKPOINT_FILE
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                domain_epoch=epoch_number,
                step_in_epoch=steps_per_epoch,
                domain_global_step=(
                    domain_global_step
                ),
                epoch_complete=True,
                epoch_loss_sum=epoch_loss_sum,
                epoch_steps_completed=(
                    epoch_steps_completed
                ),
                domain_validation_loss=(
                    domain_validation_loss
                ),
                wikitext_validation_loss=(
                    wikitext_validation_loss
                ),
                best_domain_validation_loss=(
                    best_domain_validation_loss
                ),
                base_checkpoint=base_checkpoint,
            )

            print(
                "New best domain checkpoint saved."
            )

        epoch_checkpoint_path = (
            CHECKPOINT_DIRECTORY
            / f"epoch_{epoch_number}.pt"
        )

        save_checkpoint(
            checkpoint_path=(
                epoch_checkpoint_path
            ),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            domain_epoch=epoch_number,
            step_in_epoch=steps_per_epoch,
            domain_global_step=(
                domain_global_step
            ),
            epoch_complete=True,
            epoch_loss_sum=epoch_loss_sum,
            epoch_steps_completed=(
                epoch_steps_completed
            ),
            domain_validation_loss=(
                domain_validation_loss
            ),
            wikitext_validation_loss=(
                wikitext_validation_loss
            ),
            best_domain_validation_loss=(
                best_domain_validation_loss
            ),
            base_checkpoint=base_checkpoint,
        )

        save_checkpoint(
            checkpoint_path=(
                LATEST_CHECKPOINT_FILE
            ),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            domain_epoch=epoch_number,
            step_in_epoch=steps_per_epoch,
            domain_global_step=(
                domain_global_step
            ),
            epoch_complete=True,
            epoch_loss_sum=epoch_loss_sum,
            epoch_steps_completed=(
                epoch_steps_completed
            ),
            domain_validation_loss=(
                domain_validation_loss
            ),
            wikitext_validation_loss=(
                wikitext_validation_loss
            ),
            best_domain_validation_loss=(
                best_domain_validation_loss
            ),
            base_checkpoint=base_checkpoint,
        )

        print(
            f"Epoch checkpoint saved: "
            f"{epoch_checkpoint_path}"
        )

        resume_step = 0
        epoch_loss_sum = 0.0
        epoch_steps_completed = 0

    print()
    print(
        "Domain continued pretraining "
        "completed successfully."
    )


if __name__ == "__main__":
    main()
