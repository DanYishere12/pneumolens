"""One pre-specified fine-tuning run; uses only training and validation scores."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import XRayDataset, read_manifest
from .metrics import binary_metrics
from .model import load_checkpoint
from .train import device_for


def configure_trainable(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("layer4.", "fc.")))
    # Fixed running statistics; eval does not disable autograd.
    model.eval()


def validation(model, loader, device):
    total, labels, scores = 0.0, [], []
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device))
            total += nn.functional.cross_entropy(logits, y.to(device), reduction="sum").item()
            labels.extend(y.tolist())
            scores.extend(logits.softmax(1)[:, 1].cpu().tolist())
    return total / len(labels), binary_metrics(labels, scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/baseline/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/finetuned"))
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if (args.output / "training.json").exists():
        parser.error("Output already has a completed run; use a new output directory.")
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model, source = load_checkpoint(args.checkpoint)
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != source["manifest_sha256"]:
        raise ValueError("The manifest must match the baseline's split.")
    print("Auditing data before the experiment…", flush=True)
    rows = read_manifest(args.manifest)
    train = XRayDataset(rows, "train")
    val = XRayDataset(rows, "val")
    loader = DataLoader(train, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val, batch_size=32)
    device = device_for(args.device)
    model.to(device)
    configure_trainable(model)
    counts = np.bincount([0 if r["label"] == "NORMAL" else 1 for r in train.rows], minlength=2)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(len(train) / (2 * counts), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW([{"params": model.layer4.parameters(), "lr": 1e-5},
                                  {"params": model.fc.parameters(), "lr": 1e-4}], weight_decay=1e-3)
    best, initial_metrics = validation(model, val_loader, device)
    print(f"Using {device}. Initial validation loss {best:.4f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    best_epoch, stale, history = 0, 0, []
    def save(epoch, metrics):
        checkpoint = {k: v for k, v in source.items() if k != "state_dict"}
        checkpoint.update({"training_mode": "layer4 + head fine-tuning; fixed BatchNorm statistics",
                           "finetuning_epochs": epoch, "trained_epochs": source["trained_epochs"] + epoch,
                           "source_checkpoint_sha256": source_hash, "validation": metrics,
                           "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}})
        torch.save(checkpoint, args.output / "model.pt")
    save(0, initial_metrics)
    for epoch in range(1, 9):
        start = time.monotonic()
        total = 0.0
        for index, (x, y) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            if index % 40 == 0:
                print(f"Epoch {epoch}, batch {index + 1}/{len(loader)}", flush=True)
        val_loss, metrics = validation(model, val_loader, device)
        history.append({"epoch": epoch, "weighted_train_loss": total / len(train), "validation_loss": val_loss,
                        "validation": metrics, "seconds": round(time.monotonic() - start, 2)})
        print(f"Epoch {epoch}: val loss={val_loss:.4f}, accuracy={metrics['accuracy']:.4f}", flush=True)
        if val_loss < best - 1e-5:
            best, stale, best_epoch = val_loss, 0, epoch
            save(epoch, metrics)
        else:
            stale += 1
        # Persist progress each epoch so interrupted work is inspectable.
        report = {"seed": 42, "device": str(device), "torch_version": str(torch.__version__),
                  "selected_epoch": best_epoch, "source_checkpoint_sha256": source_hash,
                  "initial_validation": initial_metrics, "history": history,
                  "split_counts": {"train": len(train), "val": len(val), "test": sum(r["split"] == "test" for r in rows)},
                  "protocol": "docs/finetuning-protocol.md", "complete": False}
        (args.output / "progress.json").write_text(json.dumps(report, indent=2))
        if stale >= 3:
            break
    report["complete"] = True
    (args.output / "training.json").write_text(json.dumps(report, indent=2))
    print(f"Finished. Selected fine-tuning epoch {best_epoch}.", flush=True)


if __name__ == "__main__":
    main()
