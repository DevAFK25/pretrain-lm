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
    / "tokenized"
    / "wikitext_train.pt"
)

VALIDATION_TOKEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "tokenized"
    / "wikitext_validation.pt"
)

TOKENIZER_FILE = (
    PROJECT_ROOT
    / "tokenizer"
    / "output"
    / "tokenizer.json"
)

CHECKPOINT_DIRECTORY = (
    PROJECT_ROOT
    / "checkpoints"
    / "wikitext_v2"
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
    # Model architecture
    vocab_size: int = 24_000
    block_size: int = 256
    embedding_dim: int = 384
    num_heads: int = 6
    num_layers: int = 8
    feed_forward_dim: int = 1_536

    # Optimisation
    batch_size: int = 32
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    weight_decay: float = 0.01

    # Training duration
    num_epochs: int = 5

    # Keep this at 200 for the final integration test.
    # Change it to None for the real full-epoch run.
    max_train_steps_per_epoch: int | None = None

    # Save during training approximately every few minutes.
    checkpoint_every_steps: int = 500

    log_every: int = 10
    random_seed: int = 42

    # Text generation
    generation_prompt: str = "The"
    generation_tokens: int = 80
    temperature: float = 0.8
    top_k: int = 40

    end_of_text_token_id: int = 0


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def load_token_ids(token_file: Path) -> torch.Tensor:
    if not token_file.exists():
        raise FileNotFoundError(
            f"Token file not found:\n{token_file}"
        )

    token_ids = torch.load(
        token_file,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(token_ids, torch.Tensor):
        raise TypeError(
            "The loaded corpus must be a torch.Tensor."
        )

    if token_ids.ndim != 1:
        raise ValueError(
            "The tokenized corpus must be one-dimensional. "
            f"Received shape: {tuple(token_ids.shape)}"
        )

    if token_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError(
            "Token IDs must use torch.int32 or torch.int64. "
            f"Received dtype: {token_ids.dtype}"
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
        # Recreating the same epoch produces the same shuffled order.
        # This lets us resume from the middle of an epoch.
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


def save_checkpoint(
    checkpoint_path: Path,
    model: SmallLanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainingConfig,
    epoch_number: int,
    step_in_epoch: int,
    global_step: int,
    epoch_complete: bool,
    epoch_loss_sum: float,
    epoch_steps_completed: int,
    validation_loss: float | None,
    best_validation_loss: float,
) -> None:
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch_number,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "epoch_complete": epoch_complete,
        "epoch_loss_sum": epoch_loss_sum,
        "epoch_steps_completed": epoch_steps_completed,
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": asdict(config),
    }

    temporary_path = checkpoint_path.with_suffix(".tmp")

    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: SmallLanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> dict | None:
    if not checkpoint_path.exists():
        return None

    print()
    print(f"Loading checkpoint: {checkpoint_path}")

    
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler_state = checkpoint.get(
        "scheduler_state_dict"
    )

    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    print(
        f"Checkpoint loaded at global step "
        f"{checkpoint['global_step']:,}."
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

    print("\nRunning validation...")

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

    prompt_ids = tokenizer.encode(prompt).ids

    if not prompt_ids:
        raise ValueError(
            "The generation prompt produced no tokens."
        )

    generated_ids = list(prompt_ids)

    for _ in range(config.generation_tokens):
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

        next_token_id = int(next_token.item())

        if (
            next_token_id
            == config.end_of_text_token_id
        ):
            break

        generated_ids.append(next_token_id)

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
    global_step: int,
    resume_step: int,
    epoch_loss_sum: float,
    epoch_steps_completed: int,
    best_validation_loss: float,
) -> tuple[float, int, float, int]:
    model.train()

    steps_per_epoch = get_steps_per_epoch(
        data_loader=data_loader,
        config=config,
    )

    total_tokens = 0

    synchronize_device(device)
    start_time = time.perf_counter()

    if resume_step > 0:
        print(
            f"Resuming epoch {epoch_number} after "
            f"step {resume_step:,}."
        )

    for batch_number, (input_ids, target_ids) in enumerate(
        data_loader,
        start=1,
    ):
        if batch_number <= resume_step:
            continue

        if batch_number > steps_per_epoch:
            break

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits, _ = model(input_ids)

        loss = calculate_loss(
            logits=logits,
            target_ids=target_ids,
            vocab_size=config.vocab_size,
            loss_function=loss_function,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch_number}, "
                f"step {batch_number}."
            )

        loss.backward()
        optimizer.step()
        scheduler.step()

        global_step += 1
        epoch_steps_completed += 1
        epoch_loss_sum += loss.item()
        total_tokens += input_ids.numel()

        if (
            batch_number == 1
            or batch_number % config.log_every == 0
            or batch_number == steps_per_epoch
        ):
            synchronize_device(device)

            elapsed = time.perf_counter() - start_time
            average_loss = (
                epoch_loss_sum
                / epoch_steps_completed
            )

            current_learning_rate = (
                optimizer.param_groups[0]["lr"]
            )

            tokens_per_second = (
                total_tokens / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"Epoch {epoch_number} | "
                f"Step {batch_number:5d}/"
                f"{steps_per_epoch:5d} | "
                f"Loss {loss.item():.4f} | "
                f"Average {average_loss:.4f} | "
                f"LR {current_learning_rate:.7f} | "
                f"{tokens_per_second:,.0f} tokens/sec"
            )

        if (
            global_step
            % config.checkpoint_every_steps
            == 0
        ):
            save_checkpoint(
                checkpoint_path=LATEST_CHECKPOINT_FILE,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch_number=epoch_number,
                step_in_epoch=batch_number,
                global_step=global_step,
                epoch_complete=False,
                epoch_loss_sum=epoch_loss_sum,
                epoch_steps_completed=(
                    epoch_steps_completed
                ),
                validation_loss=None,
                best_validation_loss=(
                    best_validation_loss
                ),
            )

            print(
                f"Progress checkpoint saved at "
                f"global step {global_step:,}."
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
        global_step,
        epoch_loss_sum,
        epoch_steps_completed,
    )


def main() -> None:
    config = TrainingConfig()

    torch.manual_seed(config.random_seed)

    device = get_device()

    print(f"Using device:          {device}")
    print(f"Epochs:                {config.num_epochs}")
    print(
        f"Maximum steps/epoch:   "
        f"{config.max_train_steps_per_epoch}"
    )
    print(
        f"Checkpoint interval:   "
        f"{config.checkpoint_every_steps} steps"
    )

    train_token_ids = load_token_ids(
        TRAIN_TOKEN_FILE
    )

    validation_token_ids = load_token_ids(
        VALIDATION_TOKEN_FILE
    )

    validation_loader = create_data_loader(
        token_ids=validation_token_ids,
        config=config,
        shuffle=False,
        drop_last=False,
    )

    first_train_loader = create_data_loader(
        token_ids=train_token_ids,
        config=config,
        shuffle=True,
        drop_last=True,
        epoch_number=1,
    )

    steps_per_epoch = get_steps_per_epoch(
        data_loader=first_train_loader,
        config=config,
    )

    total_training_steps = (
        steps_per_epoch
        * config.num_epochs
    )

    model = create_model(
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

    loss_function = torch.nn.CrossEntropyLoss()

    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(
            f"Tokenizer not found:\n{TOKENIZER_FILE}"
        )

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )

    print(f"Training tokens:       {train_token_ids.numel():,}")
    print(
        f"Validation tokens:     "
        f"{validation_token_ids.numel():,}"
    )
    print(f"Steps per epoch:       {steps_per_epoch:,}")
    print(
        f"Total training steps:  "
        f"{total_training_steps:,}"
    )
    print(
        f"Model parameters:      "
        f"{count_parameters(model):,}"
    )

    checkpoint = load_checkpoint(
        checkpoint_path=LATEST_CHECKPOINT_FILE,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    start_epoch = 1
    resume_step = 0
    global_step = 0
    epoch_loss_sum = 0.0
    epoch_steps_completed = 0
    best_validation_loss = float("inf")

    if checkpoint is not None:
        global_step = checkpoint["global_step"]

        best_validation_loss = checkpoint.get(
            "best_validation_loss",
            float("inf"),
        )

        if checkpoint["epoch_complete"]:
            start_epoch = checkpoint["epoch"] + 1
        else:
            start_epoch = checkpoint["epoch"]
            resume_step = checkpoint["step_in_epoch"]
            epoch_loss_sum = checkpoint.get(
                "epoch_loss_sum",
                0.0,
            )
            epoch_steps_completed = checkpoint.get(
                "epoch_steps_completed",
                resume_step,
            )

    # Begin a fresh continuation phase after the completed 4-epoch run.
    if checkpoint is not None and checkpoint["epoch_complete"]:
        completed_epochs = checkpoint["epoch"]

        if completed_epochs < config.num_epochs:
            continuation_learning_rate = 1e-4
    
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=continuation_learning_rate,
                weight_decay=config.weight_decay,
            )
    
            remaining_training_steps = (
                steps_per_epoch
                * (config.num_epochs - completed_epochs)
            )
    
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=remaining_training_steps,
                eta_min=config.minimum_learning_rate,
            )
    
            print(
                "Starting continuation phase with fresh optimizer "
                f"and LR {continuation_learning_rate:.7f}"
            )
    
    if start_epoch > config.num_epochs:
        print()
        print("Training is already complete.")
        return

    for epoch_number in range(
        start_epoch,
        config.num_epochs + 1,
    ):
        print()
        print("=" * 65)
        print(
            f"EPOCH {epoch_number}/"
            f"{config.num_epochs}"
        )
        print("=" * 65)

        train_loader = create_data_loader(
            token_ids=train_token_ids,
            config=config,
            shuffle=True,
            drop_last=True,
            epoch_number=epoch_number,
        )

        (
            training_loss,
            global_step,
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
            global_step=global_step,
            resume_step=resume_step,
            epoch_loss_sum=epoch_loss_sum,
            epoch_steps_completed=(
                epoch_steps_completed
            ),
            best_validation_loss=(
                best_validation_loss
            ),
        )

        validation_loss = evaluate(
            model=model,
            data_loader=validation_loader,
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
        print(f"Training loss:         {training_loss:.4f}")
        print(f"Validation loss:       {validation_loss:.4f}")
        print(f"Global step:           {global_step:,}")
        print()
        print("Generated sample:")
        print("-" * 65)
        print(generated_text)
        print("-" * 65)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            save_checkpoint(
                checkpoint_path=BEST_CHECKPOINT_FILE,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch_number=epoch_number,
                step_in_epoch=steps_per_epoch,
                global_step=global_step,
                epoch_complete=True,
                epoch_loss_sum=epoch_loss_sum,
                epoch_steps_completed=(
                    epoch_steps_completed
                ),
                validation_loss=validation_loss,
                best_validation_loss=(
                    best_validation_loss
                ),
            )

            print("New best checkpoint saved.")

        epoch_checkpoint_path = (
            CHECKPOINT_DIRECTORY
            / f"epoch_{epoch_number}.pt"
        )

        save_checkpoint(
            checkpoint_path=epoch_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch_number=epoch_number,
            step_in_epoch=steps_per_epoch,
            global_step=global_step,
            epoch_complete=True,
            epoch_loss_sum=epoch_loss_sum,
            epoch_steps_completed=(
                epoch_steps_completed
            ),
            validation_loss=validation_loss,
            best_validation_loss=(
                best_validation_loss
            ),
        )

        save_checkpoint(
            checkpoint_path=LATEST_CHECKPOINT_FILE,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch_number=epoch_number,
            step_in_epoch=steps_per_epoch,
            global_step=global_step,
            epoch_complete=True,
            epoch_loss_sum=epoch_loss_sum,
            epoch_steps_completed=(
                epoch_steps_completed
            ),
            validation_loss=validation_loss,
            best_validation_loss=(
                best_validation_loss
            ),
        )

        print(
            f"Epoch checkpoint saved: "
            f"{epoch_checkpoint_path}"
        )

        # Reset these for the next epoch.
        resume_step = 0
        epoch_loss_sum = 0.0
        epoch_steps_completed = 0

    print()
    print("Training run completed successfully.")


if __name__ == "__main__":
    main()
    