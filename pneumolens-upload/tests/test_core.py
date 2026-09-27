import csv

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from pneumolens.data import read_manifest
from pneumolens.gradcam import grad_cam
from pneumolens.imaging import overlay, preprocess, read_image, unpad_cam
from pneumolens.metrics import binary_metrics
from pneumolens.model import build_model
from pneumolens.prepare import filename_group


class KnownCAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 1, bias=False)
        self.head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([1., -1.]).view(2, 1, 1, 1))
            self.head.weight.copy_(torch.eye(2))

    def forward(self, x):
        return self.head(self.conv(x).mean((2, 3)))


def test_cam_matches_analytic_localization_and_class_changes():
    model = KnownCAM().train()
    x = torch.tensor([[[[0., 2.], [-3., 0.]]]])
    _, positive = grad_cam(model, model.conv, x, 0)
    _, negative = grad_cam(model, model.conv, x, 1)
    np.testing.assert_allclose(positive, [[0, 1], [0, 0]])
    np.testing.assert_allclose(negative, [[0, 0], [1, 0]])
    assert model.training
    assert not model.conv._forward_hooks
    assert all(p.grad is None for p in model.parameters())


def test_zero_cam_and_hook_cleanup_after_failure():
    model = KnownCAM()
    _, cam = grad_cam(model, model.conv, torch.zeros(1, 1, 2, 2), 0)
    assert np.isfinite(cam).all() and not cam.any()
    with pytest.raises(ValueError):
        grad_cam(model, model.conv, torch.zeros(1, 1, 2, 2), 2)
    assert not model.conv._forward_hooks


def test_real_resnet_supports_frozen_backbone_cam():
    torch.set_num_threads(2)
    model = build_model().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    scores, cam = grad_cam(model, model.layer4[-1], torch.randn(1, 3, 224, 224), 1)
    assert scores.shape == (2,) and cam.shape == (224, 224)
    assert np.isfinite(cam).all()
    assert scores.sum() == pytest.approx(1)


def test_letterboxing_inverse_and_transparent_zero_overlay():
    image = Image.new("RGB", (400, 200), "white")
    tensor, geometry = preprocess(image)
    assert tensor.shape == (3, 224, 224)
    assert geometry == (0, 56, 224, 112)
    cam = np.zeros((224, 224), dtype=np.float32)
    cam[56:168, :112] = 1
    field = unpad_cam(cam, geometry, image.size)
    assert field.shape == (200, 400)
    assert field[:, :190].mean() == pytest.approx(1)
    assert field[:, 210:].mean() == pytest.approx(0)
    np.testing.assert_array_equal(overlay(image, np.zeros_like(field)), image)


def test_metrics_known_confusion_matrix():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.2, 0.9])
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["roc_auc"] == 0.75
    assert binary_metrics([0, 0], [0.1, 0.2])["sensitivity"] is None


def make_manifest(tmp_path):
    rows = []
    for index, (split, label) in enumerate((s, l) for s in ["train", "val", "test"] for l in ["NORMAL", "PNEUMONIA"]):
        name = f"{index}.png"
        Image.new("RGB", (8, 8), (index * 30, 0, 0)).save(tmp_path / name)
        rows.append(dict(path=name, label=label, patient_id=f"patient{index}", split=split))
    path = tmp_path / "manifest.csv"
    return path, rows


def write_manifest(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_rejects_patient_overlap(tmp_path):
    path, rows = make_manifest(tmp_path)
    write_manifest(path, rows)
    assert len(read_manifest(path)) == 6
    rows[2]["patient_id"] = rows[0]["patient_id"]
    write_manifest(path, rows)
    with pytest.raises(ValueError, match="Patient overlap"):
        read_manifest(path)


def test_manifest_rejects_duplicate_pixels_despite_different_filename(tmp_path):
    path, rows = make_manifest(tmp_path)
    Image.open(tmp_path / "0.png").save(tmp_path / "2.png")
    write_manifest(path, rows)
    with pytest.raises(ValueError, match="Duplicate image"):
        read_manifest(path)


def test_filename_groups_do_not_split_multiple_images():
    assert filename_group("person7_bacteria_11.jpeg") == filename_group("person7_virus_12.jpeg")
    assert filename_group("NORMAL2-IM-0123-0001.jpeg") == filename_group("NORMAL2-IM-0123-0002.jpeg")
    assert filename_group("IM-0123-0001.jpeg") != filename_group("NORMAL2-IM-0123-0001.jpeg")
    with pytest.raises(ValueError):
        filename_group("mystery.jpeg")


def test_high_bit_depth_rejected(tmp_path):
    path = tmp_path / "scan.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint16)).save(path)
    with pytest.raises(ValueError, match="8-bit"):
        read_image(path)
