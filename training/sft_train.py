import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim import AdamW

from config import (
    DEVICE,
    CHECKPOINT_DIR,
    GRAD_CLIP,
    SFT_BATCH_SIZE,
    SFT_LEARNING_RATE,
    SFT_MAX_ITERS,
    SFT_EVAL_INTERVAL,
    SFT_EVAL_ITERS,
    SFT_LOG_INTERVAL
)

from model.gpt import GPTModel
from training.sft_dataset import SFTDataset


def create_dataloaders():

    train_dataset = SFTDataset(split="train")
    val_dataset = SFTDataset(split="val")
    test_dataset = SFTDataset(split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=SFT_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=SFT_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=SFT_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda")
    )

    if len(train_loader) == 0:
        raise ValueError(
            "Training DataLoader is empty. "
            "Check dataset size and SFT_BATCH_SIZE."
        )

    if len(val_loader) == 0:
        raise ValueError("Validation DataLoader is empty.")

    if len(test_loader) == 0:
        raise ValueError("Test DataLoader is empty.")

    return train_loader, val_loader, test_loader


def calculate_loss(model, x, y):

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=-100
    )

    return loss


@torch.no_grad()
def evaluate(model, train_loader, val_loader):

    model.eval()

    results = {}

    for name, loader in [
        ("train", train_loader),
        ("val", val_loader)
    ]:

        losses = []

        for i, (x, y) in enumerate(loader):

            if i >= SFT_EVAL_ITERS:
                break

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            loss = calculate_loss(model, x, y)

            losses.append(loss.item())

        if not losses:
            raise ValueError(
                f"{name} dataset has no evaluation batches."
            )

        results[name] = sum(losses) / len(losses)

    model.train()

    return results


@torch.no_grad()
def evaluate_test(model, test_loader):

    model.eval()

    losses = []

    for x, y in test_loader:

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        loss = calculate_loss(model, x, y)

        losses.append(loss.item())

    if not losses:
        raise ValueError("Test dataset has no evaluation batches.")

    model.train()

    return sum(losses) / len(losses)


def save_checkpoint(
    path,
    step,
    model,
    optimizer,
    train_loss,
    val_loss
):

    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss
        },
        path
    )


def train():

    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("Creating DataLoaders...")

    train_loader, val_loader, test_loader = create_dataloaders()

    print("Training Batches:", len(train_loader))
    print("Validation Batches:", len(val_loader))
    print("Test Batches:", len(test_loader))

    print("\nLoading GPT Model...")

    model = GPTModel().to(DEVICE)

    checkpoint_path = CHECKPOINT_DIR / "best_pretrain.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded Pretraining Checkpoint")
    print("Pretraining Step:", checkpoint["step"])

    print("\nCreating Optimizer...")

    optimizer = AdamW(
        model.parameters(),
        lr=SFT_LEARNING_RATE,
        weight_decay=0.01
    )

    print("Device:", DEVICE)
    print("Batch Size:", SFT_BATCH_SIZE)
    print("Learning Rate:", SFT_LEARNING_RATE)

    print("\nStarting SFT Training...")

    best_val_loss = float("inf")
    train_iterator = iter(train_loader)

    for step in range(1, SFT_MAX_ITERS + 1):

        try:
            x, y = next(train_iterator)

        except StopIteration:
            train_iterator = iter(train_loader)
            x, y = next(train_iterator)

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        model.train()

        loss = calculate_loss(model, x, y)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        if step % SFT_LOG_INTERVAL == 0:
            print(
                f"Step {step} | "
                f"Training Loss: {loss.item():.4f}"
            )

        if (
            step % SFT_EVAL_INTERVAL == 0
            or step == SFT_MAX_ITERS
        ):

            results = evaluate(
                model,
                train_loader,
                val_loader
            )

            train_loss = results["train"]
            val_loss = results["val"]

            print("\n----------------------------------------")
            print(f"Evaluation Step {step}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")

            save_checkpoint(
                CHECKPOINT_DIR / "latest_sft.pt",
                step,
                model,
                optimizer,
                train_loss,
                val_loss
            )

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                save_checkpoint(
                    CHECKPOINT_DIR / "best_sft.pt",
                    step,
                    model,
                    optimizer,
                    train_loss,
                    val_loss
                )

                print("Best Checkpoint Updated")

            print("----------------------------------------")

    print("\nSFT Training Finished")

    print("\nLoading Best SFT Checkpoint...")

    best_checkpoint_path = CHECKPOINT_DIR / "best_sft.pt"

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    print("Loaded Best SFT Checkpoint")
    print("Best SFT Step:", best_checkpoint["step"])

    print("\nEvaluating Test Dataset...")

    test_loss = evaluate_test(
        model,
        test_loader
    )

    print("\n----------------------------------------")
    print("Final Test Results")
    print(f"Best SFT Step: {best_checkpoint['step']}")
    print(f"Best Validation Loss: {best_checkpoint['val_loss']:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("----------------------------------------")


if __name__ == "__main__":
    train()
