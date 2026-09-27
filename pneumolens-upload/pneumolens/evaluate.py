import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import XRayDataset, read_manifest
from .metrics import binary_metrics
from .model import load_checkpoint
from .train import device_for


def main():
    parser = argparse.ArgumentParser(description="Evaluate a selected checkpoint on the untouched test split.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--test-reused", action="store_true", help="Disclose evaluation on an already examined test set.")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, metadata = load_checkpoint(args.checkpoint)
    if metadata.get("source_checkpoint_sha256") and not args.test_reused:
        parser.error("Fine-tuning follows the evaluated baseline. Pass --test-reused to disclose this comparison.")
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != metadata["manifest_sha256"]:
        raise ValueError("Manifest differs from the training manifest; use the original frozen split.")
    dataset = XRayDataset(read_manifest(args.manifest), "test")
    device = device_for(args.device)
    model.to(device)
    labels, scores = [], []
    with torch.inference_mode():
        for x, y in DataLoader(dataset, batch_size=32):
            scores.extend(model(x.to(device)).softmax(1)[:, 1].cpu().tolist())
            labels.extend(y.tolist())
    metrics = binary_metrics(labels, scores, metadata["threshold"])
    metrics["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    metrics["manifest_sha256"] = metadata["manifest_sha256"]
    metrics["notes"] = "Single-dataset image-level evaluation; scores are not calibrated clinical probabilities."
    metrics["test_set_reused"] = args.test_reused
    if args.test_reused:
        metrics["notes"] += " Exploratory comparison on the test partition already examined for the baseline; not independent validation."
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (args.output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "patient_id", "label", "pneumonia_score", "predicted_label", "correct"])
        for row, label, score in zip(dataset.rows, labels, scores):
            prediction = int(score >= metadata["threshold"])
            writer.writerow([row["path"], row["patient_id"], label, score, prediction, label == prediction])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
