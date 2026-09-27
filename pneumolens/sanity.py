"""Small, descriptive parameter-randomization check on development images only."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .gradcam import grad_cam
from .imaging import overlay, preprocess, read_image, unpad_cam
from .model import CLASSES, load_checkpoint


def comparison(reference, candidate):
    a, b = reference.ravel(), candidate.ravel()
    correlation = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-8 and b.std() > 1e-8 else None
    return {"pearson_correlation": correlation,
            "mean_absolute_change": float(np.abs(a - b).mean()),
            "positive_map": bool(b.max() > 1e-6)}


def border_fraction(field):
    """Fraction of positive CAM mass in the outer 10% on each image edge."""
    h, w = field.shape
    y, x = max(1, round(h * .1)), max(1, round(w * .1))
    border = np.ones_like(field, dtype=bool)
    border[y:h-y, x:w-x] = False
    mass = float(field.sum())
    return float(field[border].sum() / mass) if mass > 1e-8 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/finetuned/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("docs/sanity"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Selection rule is independent of scores and appearance; no test images.
    chosen = [row for label in CLASSES for row in
              sorted((r for r in rows if r["split"] == "val" and r["label"] == label),
                     key=lambda r: Path(r["path"]).name)[:2]]
    if len(chosen) != 4:
        raise ValueError("Expected at least two validation images of each class.")
    trained, _ = load_checkpoint(args.checkpoint)
    torch.manual_seed(2026)
    head = copy.deepcopy(trained)
    head.fc.reset_parameters()
    last_block = copy.deepcopy(head)
    for module in last_block.layer4.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
            module.reset_parameters()
    variants = [("Trained", trained), ("Random head", head), ("Random head + layer4", last_block)]
    args.output.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (1200, 1260), "#0c1013")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=17)
    draw.text((20, 15), "Parameter randomization / four validation images / PNEUMONIA class", font=font, fill="#e9f0f1")
    for col, name in enumerate(["Original", *(name for name, _ in variants)]):
        draw.text((20 + col * 300, 52), name, font=font, fill="#8be3cc")
    records = []
    for row_index, row in enumerate(chosen):
        path = (args.manifest.resolve().parent / row["path"]).resolve()
        image = read_image(path)
        tensor, geometry = preprocess(image)
        display = image.copy()
        display.thumbnail((272, 220))
        y = 88 + row_index * 282
        sheet.paste(display, (20 + (272 - display.width) // 2, y + (220 - display.height) // 2))
        draw.text((20, y + 223), row["label"], font=font, fill="#e9f0f1")
        draw.text((20, y + 245), path.name[:30], font=ImageFont.load_default(size=12), fill="#8c9b9f")
        record = {"filename": path.name, "label": row["label"], "split": "val", "variants": {}}
        reference = None
        for col, (name, model) in enumerate(variants, 1):
            scores, cam = grad_cam(model, model.layer4[-1], tensor[None], 1)
            # Compare unpadded content on a fixed analysis grid, not colored pixels.
            field = unpad_cam(cam, geometry, (224, 224))
            if reference is None:
                reference = field
            measures = comparison(reference, field)
            measures.update(pneumonia_score=float(scores[1]), border_mass_fraction=border_fraction(field))
            record["variants"][name] = measures
            rendered = overlay(display, unpad_cam(cam, geometry, display.size))
            sheet.paste(rendered, (20 + col * 300 + (272 - display.width) // 2, y + (220 - display.height) // 2))
            rho = measures["pearson_correlation"]
            text = f"r = {rho:.3f}" if rho is not None else "r undefined (constant map)"
            draw.text((20 + col * 300, y + 223), text, font=ImageFont.load_default(size=14), fill="#e9f0f1")
            draw.text((20 + col * 300, y + 245), f"CAM change = {measures['mean_absolute_change']:.3f}", font=ImageFont.load_default(size=14), fill="#8c9b9f")
        records.append(record)
    draw.text((20, 1220), "Kermany, Zhang & Goldbaum / CC BY 4.0 / Modified images. This is not localization validation.", font=ImageFont.load_default(size=14), fill="#8c9b9f")
    sheet.save(args.output / "randomization.png")
    report = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "selection": "First two validation filenames per class, sorted lexicographically; no outcome selection.",
        "seed": 2026, "target": "PNEUMONIA", "n": 4,
        "protocol": "CPU. Reset fc once, then cumulatively reset layer4 Conv2d and BatchNorm2d parameters and running statistics. Use PyTorch reset_parameters defaults; earlier layers stay trained. Every map is normalized separately. Compare unpadded CAMs on 224x224 grid with Pearson correlation and mean absolute difference.",
        "border_definition": "Outer 10% of width/height; approximately 36% of image area. Descriptive positive CAM mass fraction, not an anatomical lung mask or artifact detector. Null for zero maps.",
        "limitations": "Four development images and one random seed. Parameter sensitivity is not proof of faithfulness or medical localization. No random-label training, full-network randomization, or localization annotations. No pass/fail claim.",
        "reference": "https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html",
        "image_source": "https://data.mendeley.com/datasets/rscbjbr9sj/2", "license": "CC BY 4.0; overlays are modified images",
        "cases": records,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
