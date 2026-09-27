import argparse
import base64
import binascii
import csv
import hashlib
import io
import json
import mimetypes
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

from .gradcam import grad_cam
from .imaging import png_bytes, preprocess, read_image, unpad_cam
from .model import CLASSES, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "baseline": ("Frozen baseline", "artifacts/baseline", "artifacts/evaluation"),
    "finetuned": ("Fine-tuned ResNet-18", "artifacts/finetuned", "artifacts/finetuned-evaluation"),
}
OUTCOMES = {(0, 0): "tn", (0, 1): "fp", (1, 0): "fn", (1, 1): "tp"}


def data_url(image):
    return "data:image/png;base64," + base64.b64encode(png_bytes(image)).decode("ascii")


class Explorer:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.models = {}

    @lru_cache(maxsize=4)
    def _bundle(self, model_id, checkpoint_mtime, metrics_mtime, predictions_mtime):
        name, folder, evaluation = MODELS[model_id]
        checkpoint = self.root / folder / "model.pt"
        metrics = json.loads((self.root / evaluation / "metrics.json").read_text())
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != metrics.get("checkpoint_sha256"):
            raise ValueError("Evaluation does not match this checkpoint. Rerun evaluation.")
        with (self.root / evaluation / "predictions.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        cases = []
        data_root = (self.root / "data/chest_xray").resolve()
        for row in rows:
            path = Path(row["path"]).resolve()
            # Keep public IDs portable even when the source manifest uses absolute paths.
            if not path.is_relative_to(data_root):
                raise ValueError("Case image is outside the project's dataset directory.")
            relative = path.relative_to(data_root).as_posix()
            label, predicted = int(row["label"]), int(row["predicted_label"])
            cases.append({"id": hashlib.sha256(relative.encode()).hexdigest()[:16],
                          "filename": path.name, "label": CLASSES[label], "prediction": CLASSES[predicted],
                          "score": float(row["pneumonia_score"]), "outcome": OUTCOMES[label, predicted], "path": path})
        if len(cases) != metrics["n"]:
            raise ValueError("Evaluation count does not match the case catalogue.")
        observed = [[0, 0], [0, 0]]
        for case in cases:
            observed[CLASSES.index(case["label"])][CLASSES.index(case["prediction"])] += 1
        if metrics.get("confusion_matrix") is not None and observed != metrics["confusion_matrix"]:
            raise ValueError("Case outcomes do not match the evaluation confusion matrix.")
        training_path = self.root / folder / "training.json"
        training = json.loads(training_path.read_text()) if training_path.exists() else {}
        return {"id": model_id, "name": name, "metrics": metrics, "training": training,
                "cases": cases, "checkpoint": checkpoint, "digest": digest}

    def bundle(self, model_id):
        if not isinstance(model_id, str) or model_id not in MODELS:
            raise ValueError("Unknown model.")
        _, folder, evaluation = MODELS[model_id]
        paths = [self.root / folder / "model.pt", self.root / evaluation / "metrics.json",
                 self.root / evaluation / "predictions.csv"]
        if not all(p.exists() for p in paths):
            raise ValueError("This model has not completed training and evaluation.")
        return self._bundle(model_id, *(p.stat().st_mtime_ns for p in paths))

    def metadata(self):
        models, unavailable = [], []
        for model_id in MODELS:
            try:
                bundle = self.bundle(model_id)
                summary = {k: bundle[k] for k in ["id", "name", "metrics", "training", "digest"]}
                summary["counts"] = {outcome: sum(c["outcome"] == outcome for c in bundle["cases"]) for outcome in ["tn", "fp", "fn", "tp"]}
                models.append(summary)
            except (ValueError, OSError) as error:
                unavailable.append({"id": model_id, "reason": str(error)})
        return {"models": models, "unavailable": unavailable,
                "sanity_available": (self.root / "docs/sanity/results.json").exists() and (self.root / "docs/sanity/randomization.png").exists(),
                "default_model": "finetuned" if any(m["id"] == "finetuned" and m["training"].get("selected_epoch", 0) > 0 for m in models) else models[0]["id"] if models else None}

    def cases(self, model_id, outcome="all", offset=0, limit=12):
        if outcome not in {"all", "tn", "fp", "fn", "tp"}:
            raise ValueError("Unknown outcome filter.")
        if offset < 0 or not 1 <= limit <= 48:
            raise ValueError("Invalid pagination.")
        rows = [c for c in self.bundle(model_id)["cases"] if outcome == "all" or c["outcome"] == outcome]
        return {"total": len(rows), "offset": offset, "cases": [{k: v for k, v in c.items() if k != "path"} for c in rows[offset:offset + limit]]}

    def case(self, model_id, case_id):
        for case in self.bundle(model_id)["cases"]:
            if case["id"] == case_id:
                return case
        raise ValueError("Unknown example.")

    @lru_cache(maxsize=128)
    def thumbnail(self, path, modified_ns):
        image = read_image(path)
        image.thumbnail((240, 200))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=78)
        return output.getvalue()

    def analyze(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")
        model_id, target = payload.get("model", "baseline"), payload.get("target", "PNEUMONIA")
        if target not in CLASSES:
            raise ValueError("Explain class must be NORMAL or PNEUMONIA.")
        if bool(payload.get("case_id")) == bool(payload.get("image")):
            raise ValueError("Choose exactly one dataset example or uploaded image.")
        bundle = self.bundle(model_id)
        if payload.get("case_id"):
            case = self.case(model_id, payload["case_id"])
            image = read_image(case["path"])
            filename, label = case["filename"], case["label"]
        else:
            try:
                raw = base64.b64decode(payload["image"], validate=True)
            except (binascii.Error, TypeError) as error:
                raise ValueError("The uploaded image could not be decoded.") from error
            if len(raw) > 10 * 1024 * 1024:
                raise ValueError("Choose an image under 10 MB.")
            image = read_image(io.BytesIO(raw))
            filename, label = str(payload.get("filename", "Uploaded image"))[:160], None
        tensor, geometry = preprocess(image)
        if not self.lock.acquire(timeout=0.2):
            raise BlockingIOError("Another image is being analyzed. Please try again in a moment.")
        try:
            if model_id not in self.models or self.models[model_id][0] != bundle["digest"]:
                model, metadata = load_checkpoint(bundle["checkpoint"])
                self.models[model_id] = (bundle["digest"], model, metadata)
            _, model, metadata = self.models[model_id]
            scores, cam = grad_cam(model, model.layer4[-1], tensor[None], CLASSES.index(target))
        finally:
            self.lock.release()
        original_size = image.size
        image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        field = unpad_cam(cam, geometry, image.size)
        return {"model": model_id, "model_name": bundle["name"], "checkpoint_sha256": bundle["digest"],
                "filename": filename, "label": label, "target": target,
                "prediction": CLASSES[int(scores[1] >= metadata["threshold"])],
                "score": float(scores[1]), "threshold": metadata["threshold"],
                "width": image.width, "height": image.height, "source_size": list(original_size),
                "image": data_url(image), "cam": data_url(Image.fromarray(np.round(field * 255).astype(np.uint8))),
                "has_activation": bool(np.any(field > 1e-6))}


def handler_for(explorer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Do not persist request bodies or filenames from uploads.
            pass

        def send(self, data, mime="application/json", status=200):
            if mime == "application/json":
                data = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def valid_origin(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            return self.headers.get("Host") in allowed and (not self.headers.get("Origin") or urlparse(self.headers["Origin"]).netloc in allowed)

        def do_GET(self):
            if not self.valid_origin():
                return self.send({"error": "Use the local app address."}, status=403)
            request = urlparse(self.path)
            query = parse_qs(request.query)
            model_id = query.get("model", ["baseline"])[0]
            try:
                if request.path == "/api/meta":
                    return self.send(explorer.metadata())
                if request.path == "/api/cases":
                    return self.send(explorer.cases(model_id, query.get("outcome", ["all"])[0],
                                                    int(query.get("offset", [0])[0]), int(query.get("limit", [12])[0])))
                if request.path.startswith("/api/thumbnail/"):
                    case = explorer.case(model_id, request.path.rsplit("/", 1)[-1])
                    return self.send(explorer.thumbnail(str(case["path"]), case["path"].stat().st_mtime_ns), "image/jpeg")
                reports = {"/sanity.png": "randomization.png", "/sanity.json": "results.json"}
                if request.path in reports:
                    file = explorer.root / "docs/sanity" / reports[request.path]
                    if file.suffix == ".json":
                        return self.send(json.loads(file.read_text()))
                    return self.send(file.read_bytes(), "image/png")
                if request.path.startswith("/demo/"):
                    demo_root = (explorer.root / "web/demo").resolve()
                    file = (explorer.root / "web" / request.path.lstrip("/")).resolve()
                    if not file.is_relative_to(demo_root) or file.suffix not in {".png", ".json"}:
                        return self.send({"error": "Not found."}, status=404)
                    if file.suffix == ".json":
                        return self.send(json.loads(file.read_text()))
                    return self.send(file.read_bytes(), "image/png")
                static = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css", "/viewer.mjs": "viewer.mjs", "/recorded.mjs": "recorded.mjs"}
                if request.path in static:
                    file = explorer.root / "web" / static[request.path]
                    mime = "text/javascript" if file.suffix in {".js", ".mjs"} else mimetypes.guess_type(file.name)[0]
                    return self.send(file.read_bytes(), mime)
                return self.send({"error": "Not found."}, status=404)
            except (ValueError, OSError) as error:
                return self.send({"error": str(error)}, status=400)

        def do_POST(self):
            if not self.valid_origin():
                return self.send({"error": "Use the local app address."}, status=403)
            if self.path != "/api/analyze":
                return self.send({"error": "Not found."}, status=404)
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send({"error": "Expected application/json."}, status=415)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 14 * 1024 * 1024:
                    return self.send({"error": "Image request exceeds the 10 MB upload limit."}, status=413)
                self.connection.settimeout(20)
                payload = json.loads(self.rfile.read(length))
                return self.send(explorer.analyze(payload))
            except BlockingIOError as error:
                return self.send({"error": str(error)}, status=503)
            except (ValueError, OSError, RuntimeError, Image.DecompressionBombError) as error:
                return self.send({"error": str(error)}, status=400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Run the local PneumoLens explorer.")
    parser.add_argument("--port", type=int, default=8502)
    args = parser.parse_args()
    torch.set_num_threads(4)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(Explorer()))
    server.daemon_threads = True
    print(f"PneumoLens: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
