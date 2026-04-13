import argparse
import os
import traceback

import torch
import wandb

from data import get_data_loaders
from model import build_model
from config_ops import load_yaml, flatten_dict, apply_overrides

import multiprocessing


def log_run_exception(run, stage, exc, device=None, epoch=None, step=None):
    error_type = type(exc).__name__
    error_message = repr(exc)
    stack_trace = traceback.format_exc()

    print(f"[ERROR] stage={stage} type={error_type} message={error_message}")
    print(stack_trace)

    if device is not None and str(device).startswith("cuda"):
        try:
            print(torch.cuda.memory_summary(device=device, abbreviated=True))
        except Exception:
            pass

    if run is None:
        return

    payload = {
        "error/stage": stage,
        "error/type": error_type,
        "error/message": error_message,
        "error/pid": os.getpid(),
        "error/cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if epoch is not None:
        payload["error/epoch"] = epoch
    if step is not None:
        payload["error/step"] = step

    try:
        run.log(payload)
        run.summary["error_type"] = error_type
        run.summary["error_stage"] = stage
        run.summary["error_message"] = error_message[:500]
        run.summary["error_traceback"] = stack_trace[-4000:]
    except Exception:
        pass


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def build_optimizer(model, optimizer_config):
    name = optimizer_config["name"].lower()
    lr = optimizer_config["lr"]
    weight_decay = optimizer_config.get("weight_decay", 0.0)

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = optimizer_config.get("momentum", 0.9)
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")


def train_one_run(config, run):
    requested_device = config["train"].get("device", "auto")
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_config = config["data"]
    train_loader, val_loader, test_loader = get_data_loaders(
        val_ratio=data_config["val_ratio"],
        test_ratio=data_config["test_ratio"],
        img_size=tuple(data_config["img_size"]),
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
    )

    model = build_model(
        name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config["optimizer"])

    if config["wandb"].get("watch_model", False):
        run.watch(model, log="all", log_freq=100)

    best_val_acc = 0.0
    epochs = config["train"]["epochs"]
    try:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            total = 0

            for step, (images, labels) in enumerate(train_loader):
                try:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    logits = model(images)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    total += images.size(0)
                except Exception as exc:
                    log_run_exception(
                        run,
                        stage="train_step",
                        exc=exc,
                        device=device,
                        epoch=epoch + 1,
                        step=step,
                    )
                    raise

            train_loss = running_loss / total
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            best_val_acc = max(best_val_acc, val_acc)

            run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/best_acc": best_val_acc,
                }
            )
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        run.log({"test/loss": test_loss, "test/acc": test_acc})
        print(f"Test | loss={test_loss:.4f} acc={test_acc:.4f}")
    except Exception as exc:
        log_run_exception(run, stage="train_run", exc=exc, device=device)
        raise


def run_single(config):
    with wandb.init(
        entity=config["wandb"].get("entity"),
        project=config["wandb"]["project"],
        name=config["wandb"].get("run_name"),
        mode=config["wandb"].get("mode", "online"),
        config=flatten_dict(config),
    ) as run:
        effective_config = apply_overrides(config, dict(run.config))
        train_one_run(effective_config, run)


def run_sweep(config, sweep_config, count):
    project = config["wandb"]["project"]
    entity = config["wandb"].get("entity")
    sweep_project = sweep_config.get("project", project)

    sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_project, entity=entity)

    def sweep_train():
        with wandb.init(
            entity=entity,
            project=sweep_project,
            mode=config["wandb"].get("mode", "online"),
            config=flatten_dict(config),
        ) as run:
            effective_config = apply_overrides(config, dict(run.config))
            run.config.update(flatten_dict(effective_config), allow_val_change=True)
            try:
                train_one_run(effective_config, run)
            except Exception as exc:
                device_name = effective_config.get("train", {}).get("device", "auto")
                log_run_exception(run, stage="sweep_train", exc=exc, device=device_name)
                raise

    def start_agent(gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        wandb.agent(sweep_id, function=sweep_train, count=count)

    gpu_ids = [0, 1]
    processes = []

    for gpu_id in gpu_ids:
        p = multiprocessing.Process(target=start_agent, args=(gpu_id,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def parse_args():
    parser = argparse.ArgumentParser(description="Train MiniPetDetectorCOV models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lenet5_base.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Path to W&B sweep config YAML",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of sweep runs for this agent",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml(args.config)

    if args.sweep:
        sweep_config = load_yaml(args.sweep)
        run_sweep(config, sweep_config, count=args.count)
    else:
        run_single(config)


if __name__ == "__main__":
    main()
