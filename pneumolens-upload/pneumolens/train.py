"""Train a reproducible transfer-learning baseline; never score the test split here."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import XRayDataset, read_manifest
from .metrics import binary_metrics
from .model import CLASSES, PREPROCESS, build_model


def device_for(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def extract_features(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features, labels = [], []
    with torch.inference_mode():
        for index, (x, y) in enumerate(loader):
            features.append(model(x.to(device)).cpu())
            labels.append(y)
            if index % 20 == 0:
                print(f"Extracted batch {index + 1}/{len(loader)}", flush=True)
    return torch.cat(features), torch.cat(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    rows = read_manifest(args.manifest)
    device = device_for(args.device)
    print(f"Using {device}. Audited {len(rows)} images. Test images are not scored.", flush=True)
    model = build_model(pretrained=True).to(device).eval()
    head = model.fc.cpu()
    model.fc = nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_x, train_y = extract_features(model, XRayDataset(rows, "train"), device, args.batch_size)
    val_x, val_y = extract_features(model, XRayDataset(rows, "val"), device, args.batch_size)
    # Optimize the linear head on CPU; cached frozen features avoid repeated backbone passes.
    counts = torch.bincount(train_y, minlength=2).float()
    criterion = nn.CrossEntropyLoss(weight=len(train_y) / (2 * counts))
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    args.output.mkdir(parents=True, exist_ok=True)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        head.train()
        train_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        head.eval()
        with torch.inference_mode():
            logits = head(val_x)
            val_loss = nn.functional.cross_entropy(logits, val_y).item()
            metrics = binary_metrics(val_y.numpy(), logits.softmax(1)[:, 1].numpy())
        history.append({"epoch": epoch, "weighted_train_loss": train_loss / len(train_y),
                        "validation_loss": val_loss, "validation": metrics})
        print(f"Epoch {epoch}: val loss={val_loss:.4f}, AUROC={metrics['roc_auc']:.4f}", flush=True)
        if val_loss < best - 1e-5:
            best, stale = val_loss, 0
            model.fc = head
            checkpoint = {
                "format_version": 1, "architecture": "resnet18", "classes": CLASSES,
                "preprocess": PREPROCESS, "threshold": 0.5, "trained_epochs": epoch,
                "training_mode": "frozen ImageNet backbone, class-weighted linear head",
                "seed": args.seed, "validation": metrics,
                "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            torch.save(checkpoint, args.output / "model.pt")
        else:
            stale += 1
        if stale >= 8:
            break
    report = {"seed": args.seed, "device": str(device), "torch_version": str(torch.__version__),
              "selected_epoch": checkpoint["trained_epochs"],
              "split_counts": {s: sum(r["split"] == s for r in rows) for s in ["train", "val", "test"]},
              "history": history}
    (args.output / "training.json").write_text(json.dumps(report, indent=2))
    print(f"Saved {args.output / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
