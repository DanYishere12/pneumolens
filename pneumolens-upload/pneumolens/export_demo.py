"""Export a small, explicitly recorded demo from evaluated local checkpoints."""
import base64
import hashlib
import json
from pathlib import Path

def main():
    print("Loading model libraries for the recorded export…", flush=True)
    import torch
    from .model import CLASSES
    from .server import Explorer, ROOT

    torch.set_num_threads(4)
    explorer = Explorer()
    metadata = explorer.metadata()
    if not metadata["models"]:
        raise ValueError("Train and evaluate a model before exporting a recorded demo.")
    # Union keeps the same scans selectable when switching models. Selection is
    # by outcome and manifest order, never by how attractive a heatmap looks.
    selected = set()
    for model in metadata["models"]:
        for outcome in ("tp", "tn", "fp", "fn"):
            cases = explorer.cases(model["id"], outcome, limit=1)["cases"]
            selected.update(case["id"] for case in cases)
    output = ROOT / "web/demo"
    (output / "assets").mkdir(parents=True, exist_ok=True)
    package = {"schema_version": 1, "recorded": True,
               "selection": "Union of the first example of each outcome per evaluated model, in test manifest order. This selected gallery is not a representative performance sample.",
               "source": "https://data.mendeley.com/datasets/rscbjbr9sj/2",
               "license": "CC BY 4.0; resized images and attribution maps are modified versions.",
               "default_model": metadata["default_model"], "sanity_available": False,
               "unavailable": [], "models": [], "cases": {}, "analyses": {}, "assets_sha256": {}}
    for model in metadata["models"]:
        model_id = model["id"]
        cases = [{k: v for k, v in case.items() if k != "path"}
                 for case in explorer.bundle(model_id)["cases"] if case["id"] in selected]
        summary = {key: model[key] for key in ("id", "name", "metrics", "digest", "counts")}
        summary["gallery_counts"] = {outcome: sum(case["outcome"] == outcome for case in cases)
                                     for outcome in ("tp", "tn", "fp", "fn")}
        package["models"].append(summary)
        package["cases"][model_id] = cases
        package["analyses"][model_id] = {}
        for case in cases:
            analyses = {}
            for target in CLASSES:
                result = explorer.analyze({"model": model_id, "case_id": case["id"], "target": target})
                if abs(result["score"] - case["score"]) > 1e-5:
                    raise ValueError("Recorded inference does not match evaluated score.")
                for kind in ("image", "cam"):
                    filename = f"{case['id']}.png" if kind == "image" else f"{case['id']}-{model_id}-{target.lower()}.png"
                    relative = f"assets/{filename}"
                    raw = base64.b64decode(result[kind].split(",", 1)[1])
                    (output / relative).write_bytes(raw)
                    package["assets_sha256"][relative] = hashlib.sha256(raw).hexdigest()
                    result[kind] = f"/demo/{relative}"
                result["recorded"] = True
                analyses[target] = result
            case["thumbnail"] = analyses["PNEUMONIA"]["image"]
            package["analyses"][model_id][case["id"]] = analyses
    (output / "manifest.json").write_text(json.dumps(package, indent=2, allow_nan=False) + "\n")
    print(f"Exported {len(selected)} selected scans, {len(package['models'])} models, both explanation classes.")


if __name__ == "__main__":
    main()
