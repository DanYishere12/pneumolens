"""Export the first test example of each outcome, without selecting attractive heatmaps."""
import csv
import json
import shutil
from pathlib import Path

import torch

from .gradcam import grad_cam
from .imaging import overlay, preprocess, read_image, unpad_cam
from .model import load_checkpoint


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    with (root / "artifacts/evaluation/predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    model, _ = load_checkpoint(root / "artifacts/baseline/model.pt")
    output = root / "docs/examples"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for truth, prediction, name, slug in [
        (1, 1, "Pneumonia · correctly classified", "true-positive"),
        (0, 0, "Normal · correctly classified", "true-negative"),
        (0, 1, "Normal · false positive", "false-positive"),
        (1, 0, "Pneumonia · false negative", "false-negative"),
    ]:
        row = next((r for r in rows if int(r["label"]) == truth and int(r["predicted_label"]) == prediction), None)
        if row is None:
            continue
        source = Path(row["path"])
        destination = output / f"{slug}{source.suffix}"
        shutil.copyfile(source, destination)
        image = read_image(source)
        tensor, geometry = preprocess(image)
        _, cam = grad_cam(model, model.layer4[-1], tensor[None], 1)
        rendered = overlay(image, unpad_cam(cam, geometry, image.size))
        rendered.save(output / f"{slug}-overlay.png")
        records.append({"name": name, "image": str(destination.relative_to(root)),
                        "label": "PNEUMONIA" if truth else "NORMAL",
                        "original_filename": source.name, "pneumonia_score": float(row["pneumonia_score"]),
                        "selection": "First matching outcome in deterministic test manifest order",
                        "source": "https://data.mendeley.com/datasets/rscbjbr9sj/2", "license": "CC BY 4.0"})
    (root / "docs/examples.json").write_text(json.dumps(records, indent=2))
    print(f"Exported {len(records)} examples with original images and pneumonia-class overlays.")


if __name__ == "__main__":
    main()
