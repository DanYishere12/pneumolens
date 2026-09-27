import csv
import hashlib
from pathlib import Path

from torch.utils.data import Dataset

from .imaging import preprocess, read_image
from .model import CLASSES


def read_manifest(path):
    """Audit split labels, patient separation, and decoded-image duplicates."""
    path = Path(path).resolve()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"path", "label", "patient_id", "split"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest needs path,label,patient_id,split columns.")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty.")
    patients, hashes, paths = {}, {}, set()
    for row in rows:
        if row["label"] not in CLASSES or row["split"] not in {"train", "val", "test"}:
            raise ValueError("Labels must be NORMAL/PNEUMONIA; splits must be train/val/test.")
        subject = row["patient_id"].strip()
        if not subject:
            raise ValueError("Every row needs a patient/group identifier; document its provenance.")
        if subject in patients and patients[subject] != row["split"]:
            raise ValueError(f"Patient overlap between splits: {subject}")
        patients[subject] = row["split"]
        image_path = (path.parent / row["path"]).resolve()
        if image_path in paths:
            raise ValueError(f"Repeated image path: {image_path.name}")
        paths.add(image_path)
        image = read_image(image_path)
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if digest in hashes and hashes[digest] != row["split"]:
            raise ValueError(f"Duplicate image across splits: {image_path.name}")
        hashes[digest] = row["split"]
        row["path"] = str(image_path)
        row["patient_id"] = subject
    for split in ("train", "val", "test"):
        if {r["label"] for r in rows if r["split"] == split} != set(CLASSES):
            raise ValueError(f"Split {split} must contain both classes.")
    return rows


class XRayDataset(Dataset):
    def __init__(self, rows, split):
        self.rows = [r for r in rows if r["split"] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        tensor, _ = preprocess(read_image(row["path"]))
        return tensor, CLASSES.index(row["label"])
