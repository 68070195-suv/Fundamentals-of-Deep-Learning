
import argparse
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset

from config import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_ITERS,
    EVAL_INTERVAL,
    EVAL_ITERS,
    LOG_INTERVAL,
    GRAD_CLIP,
    CHECKPOINT_DIR
)

from model.gpt import GPTModel
from training.dataset import PretrainingDataset


def create_dataloaders():

    dataset = PretrainingDataset()

    train_size = int(len(dataset) * 0.9)

    train_dataset = Subset(
        dataset,
        range(0, train_size)
    )

    val_dataset = Subset(
        dataset,
        range(train_size, len(dataset))
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader


def calculate_loss(model, x, y):

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1)
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
        iterator = iter(loader)

        for _ in range(min(EVAL_ITERS, len(loader))):

            try:
                x, y = next(iterator)

            except StopIteration:
                break

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            loss = calculate_loss(model, x, y)

            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()

    return results


def save_checkpoint(
    model,
    optimizer,
    step,
    train_loss,
    val_loss,
    filename
):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss
    }

    torch.save(
        checkpoint,
        CHECKPOINT_DIR / filename
    )


def train(max_iters, eval_interval):

    torch.manual_seed(42)

    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(42)

    print("Device:", DEVICE)

    # โหลดข้อมูล
    train_loader, val_loader = create_dataloaders()

    print("Training Batches:", len(train_loader))
    print("Validation Batches:", len(val_loader))

    # Create GPT
    model = GPTModel().to(DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    train_iterator = iter(train_loader)

    best_val_loss = float("inf")

    # test-pre
    losses = evaluate(
        model,
        train_loader,
        val_loader
    )

    print(
        f"Step 0 | "
        f"Train Loss: {losses['train']:.4f} | "
        f"Val Loss: {losses['val']:.4f}"
    )

    for step in range(1, max_iters + 1):

        model.train()

        try:
            x, y = next(train_iterator)

        except StopIteration:
            train_iterator = iter(train_loader)
            x, y = next(train_iterator)

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        # Loss
        loss = calculate_loss(model, x, y)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        # Gradient
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        # Parameters
        optimizer.step()

        if step % LOG_INTERVAL == 0:

            print(
                f"Step {step} | "
                f"Training Loss: {loss.item():.4f}"
            )

        if step % eval_interval == 0 or step == max_iters:

            losses = evaluate(
                model,
                train_loader,
                val_loader
            )

            train_loss = losses["train"]
            val_loss = losses["val"]

            print(
                f"\nEvaluation Step {step}"
            )

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")

            save_checkpoint(
                model,
                optimizer,
                step,
                train_loss,
                val_loss,
                "latest_pretrain.pt"
            )

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    train_loss,
                    val_loss,
                    "best_pretrain.pt"
                )

                print("Best Checkpoint Updated")

            print("-" * 40)

    print("\nPretraining Finished")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_iters",
        type=int,
        default=MAX_ITERS
    )

    parser.add_argument(
        "--eval_interval",
        type=int,
        default=EVAL_INTERVAL
    )

    args = parser.parse_args()

    train(
        max_iters=args.max_iters,
        eval_interval=args.eval_interval
    )
