import base64
import csv
import hashlib
import http.client
import io
import json
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from pneumolens import server
from pneumolens.finetune import configure_trainable
from pneumolens.imaging import png_bytes
from pneumolens.model import build_model


class ToyModel(nn.Module):
    """An analytically simple test double, never used by the app."""
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Sequential(nn.Conv2d(3, 2, 1, bias=False))
        with torch.no_grad():
            self.layer4[0].weight[0].fill_(1)
            self.layer4[0].weight[1].fill_(-1)

    def forward(self, x):
        return self.layer4(x).mean((2, 3))


@pytest.fixture
def explorer(tmp_path, monkeypatch):
    checkpoint = tmp_path / "artifacts/baseline/model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"TEST DOUBLE - NOT A TRAINED MODEL")
    evaluation = tmp_path / "artifacts/evaluation"
    evaluation.mkdir(parents=True)
    (evaluation / "metrics.json").write_text(json.dumps({"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "n": 4}))
    rows = []
    for index, (label, predicted) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        image_path = tmp_path / "data/chest_xray/test" / server.CLASSES[label] / f"image{index}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 32), (200, 200, 200)).save(image_path)
        rows.append({"path": str(image_path), "label": label, "predicted_label": predicted, "pneumonia_score": 0.8 if predicted else 0.2})
    with (evaluation / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    monkeypatch.setattr(server, "load_checkpoint", lambda path: (ToyModel().eval(), {"threshold": 0.5}))
    return server.Explorer(tmp_path)


def test_gallery_filters_paginates_and_does_not_expose_local_paths(explorer):
    assert explorer.metadata()["models"][0]["counts"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    page = explorer.cases("baseline", "fp")
    assert page["total"] == 1 and page["cases"][0]["outcome"] == "fp"
    assert "path" not in page["cases"][0]
    assert len(explorer.cases("baseline", offset=2, limit=2)["cases"]) == 2
    with pytest.raises(ValueError): explorer.cases("baseline", offset=-1)
    with pytest.raises(ValueError): explorer.case("baseline", "../../etc/passwd")
    with pytest.raises(ValueError): explorer.bundle({})


def test_checkpoint_report_mismatch_is_rejected(explorer):
    path = explorer.root / "artifacts/evaluation/metrics.json"
    path.write_text(json.dumps({"n": 4, "checkpoint_sha256": "wrong"}))
    with pytest.raises(ValueError, match="does not match"):
        explorer.bundle("baseline")


def test_stale_prediction_catalogue_is_rejected(explorer):
    path = explorer.root / "artifacts/evaluation/metrics.json"
    metrics = json.loads(path.read_text())
    metrics["confusion_matrix"] = [[2, 0], [0, 2]]
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="confusion matrix"):
        explorer.bundle("baseline")


def test_analysis_class_switch_preserves_prediction_and_returns_aligned_maps(explorer):
    case = explorer.cases("baseline")["cases"][0]
    first = explorer.analyze({"case_id": case["id"], "target": "NORMAL"})
    second = explorer.analyze({"case_id": case["id"], "target": "PNEUMONIA"})
    assert first["score"] == pytest.approx(second["score"])
    assert first["prediction"] == second["prediction"]
    assert first["cam"] != second["cam"]
    for name in ["image", "cam"]:
        image = Image.open(io.BytesIO(base64.b64decode(first[name].split(",")[1])))
        assert image.size == (first["width"], first["height"]) == (64, 32)
    assert not explorer.models["baseline"][1].layer4[-1]._forward_hooks


def test_upload_has_no_label_and_invalid_inputs_fail(explorer):
    before = {p for p in explorer.root.rglob("*")}
    encoded = base64.b64encode(png_bytes(Image.new("RGB", (32, 64), "white"))).decode()
    result = explorer.analyze({"image": encoded, "filename": "research.png"})
    assert result["label"] is None and result["source_size"] == [32, 64]
    assert {p for p in explorer.root.rglob("*")} == before
    for payload in [[], {}, {"image": "bad$"}, {"image": encoded, "case_id": "both"}, {"image": encoded, "target": "INVALID"}]:
        with pytest.raises(ValueError): explorer.analyze(payload)


def test_finetuning_leaves_earlier_layers_and_batchnorm_statistics_frozen():
    model = build_model()
    configure_trainable(model)
    assert all(p.requires_grad == name.startswith(("layer4.", "fc.")) for name, p in model.named_parameters())
    assert all(not module.training for module in model.modules() if isinstance(module, nn.BatchNorm2d))
    before = model.layer4[0].bn1.running_mean.clone()
    model(torch.randn(2, 3, 224, 224)).sum().backward()
    assert model.layer4[0].conv1.weight.grad is not None
    assert model.layer3[0].conv1.weight.grad is None
    torch.testing.assert_close(before, model.layer4[0].bn1.running_mean)


def test_http_boundaries_and_real_analysis_response(explorer):
    torch.set_num_threads(2)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.handler_for(explorer))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=15)
        connection.request("GET", "/api/cases?outcome=fn")
        response = connection.getresponse()
        assert response.status == 200
        case = json.loads(response.read())["cases"][0]
        connection.request("POST", "/api/analyze", json.dumps({"case_id": case["id"]}), {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200 and "image" in json.loads(response.read())
        connection.request("POST", "/api/analyze", "{}", {"Content-Type": "application/json", "Origin": "https://unrelated.example"})
        response = connection.getresponse()
        assert response.status == 403
        response.read()
        connection.request("GET", "/../artifacts/baseline/model.pt")
        response = connection.getresponse()
        assert response.status == 404
        response.read()
        connection.request("POST", "/api/analyze", "{}", {"Content-Type": "text/plain"})
        response = connection.getresponse()
        assert response.status == 415
        response.read(); connection.close()
    finally:
        httpd.shutdown(); httpd.server_close(); thread.join(timeout=2)
