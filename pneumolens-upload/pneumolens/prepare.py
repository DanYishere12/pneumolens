"""Prepare the Kermany/Kaggle folder export with conservative filename grouping."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from .imaging import read_image


def filename_group(name):
    match = re.match(r"(person\d+)_", name, re.I)
    if match:
        return match.group(1).lower()
    match = re.match(r"((?:NORMAL2-)?IM-\d+)-", name, re.I)
    if match:
        return match.group(1).lower()
    raise ValueError(f"Unknown filename pattern: {name}; provide verified patient metadata instead.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Folder directly containing train, val, test")
    parser.add_argument("--output", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = []
    for split in ["train", "val", "test"]:
        for label in ["NORMAL", "PNEUMONIA"]:
            folder = args.root / split / label
            if not folder.is_dir():
                raise ValueError(f"Missing dataset folder: {folder}")
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and not path.name.startswith("."):
                    image = read_image(path)
                    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                    rows.append({"path": str(path.resolve()), "label": label,
                                 "patient_id": filename_group(path.name), "split": split, "digest": digest})
    # Preserve the supplied test partition. Remove any matching group/image from development.
    test_groups = {r["patient_id"] for r in rows if r["split"] == "test"}
    test_hashes = {r["digest"] for r in rows if r["split"] == "test"}
    development = [r for r in rows if r["split"] != "test"
                   and r["patient_id"] not in test_groups and r["digest"] not in test_hashes]
    removed_overlap = len(rows) - sum(r["split"] == "test" for r in rows) - len(development)
    # Deduplicate development before splitting, also detecting inconsistent labels.
    unique = {}
    for row in development:
        if row["digest"] in unique and unique[row["digest"]]["label"] != row["label"]:
            raise ValueError("Identical image has conflicting labels.")
        unique.setdefault(row["digest"], row)
    removed_duplicates = len(development) - len(unique)
    development = list(unique.values())
    group_labels = {}
    for row in development:
        group = row["patient_id"]
        if group in group_labels and group_labels[group] != row["label"]:
            raise ValueError("Filename group contains conflicting labels; inspect source metadata.")
        group_labels[group] = row["label"]
    groups = sorted(group_labels)
    _, validation_groups = train_test_split(groups, test_size=0.2, random_state=args.seed,
                                           stratify=[group_labels[g] for g in groups])
    validation_groups = set(validation_groups)
    for row in development:
        row["split"] = "val" if row["patient_id"] in validation_groups else "train"
    final = development + [r for r in rows if r["split"] == "test"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "patient_id", "split"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final)
    audit = {"source": "https://data.mendeley.com/datasets/rscbjbr9sj/2",
             "mirror": "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
             "seed": args.seed, "removed_development_overlap": removed_overlap,
             "removed_development_duplicates": removed_duplicates,
             "counts": dict(Counter(f"{r['split']}/{r['label']}" for r in final)),
             "grouping": "Conservative filename-derived groups; true patient identity across naming families is unverified.",
             "split_method": "Preserve supplied test; pool train/val then stratify filename groups into 80/20 development split."}
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
