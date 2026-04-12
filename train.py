import torch
import wandb

from data import get_data_loaders
from model import LeNet5Like


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_data_loaders()

    run = wandb.init(
        entity="johnyboro-personal",
        project="MiniPetDetector",
        name="lenet5-like-baseline",
        config={
            "epochs": 10,
            "lr": 1e-3,
            "optimizer": "Adam",
            "architecture": "LeNet5Like",
            "num_classes": 37,
            "device": str(device),
        },
    )
    config = run.config

    model = LeNet5Like(num_classes=config.num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    wandb.watch(model, log="all", log_freq=100)

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total += images.size(0)

        train_loss = running_loss / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    wandb.log({"test_loss": test_loss, "test_acc": test_acc})
    print(f"Test | loss={test_loss:.4f} acc={test_acc:.4f}")
    wandb.finish()


if __name__ == "__main__":
    train()
