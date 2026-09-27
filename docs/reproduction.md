# Training, evaluation and implementation notes

Run all commands from the repository root, the folder containing `serve.py`. For the dependency-free recorded demo, see the [main README](../README.md).

**Look beyond the prediction.**

A chest X-ray classification and interpretation project built with PyTorch, a local Python API, and a custom JavaScript/canvas interface. Compare an original scan with a real Grad-CAM overlay, switch the class being explained, and inspect held-out evaluation results.

| Original dataset image | Pneumonia-class Grad-CAM |
| --- | --- |
| ![Original public dataset example](examples/true-positive.jpeg) | ![Actual baseline Grad-CAM overlay](examples/true-positive-overlay.png) |

Image: Kermany, Zhang & Goldbaum, CC BY 4.0; overlay is a modified version. This is the first true-positive example in test manifest order, not a localization benchmark. Broad activation, including outside the lungs, illustrates why a correct label does not establish a valid medical explanation.

PneumoLens complements **Neural Canvas** (neural networks from scratch) and **Chroma Lab** (unsupervised clustering) with transfer learning, real image data, model interpretation, and evaluation under dataset limitations.

This is an educational research demo. It is not a diagnostic tool. Grad-CAM highlights positive influence on a model's selected class logit; it does not identify the cause of pneumonia, verify disease locations, or segment infected tissue. The app cannot determine whether an uploaded image is an appropriate chest X-ray.

## Run the app

Python 3.11 is the tested environment. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python serve.py
```

Open **http://127.0.0.1:8502/**. No frontend build step is needed. The original Streamlit app remains available via `python -m streamlit run app.py --server.address 127.0.0.1` on port 8501.

The custom app reads evaluated checkpoints and prediction CSVs from `artifacts/baseline/` + `artifacts/evaluation/` and optionally `artifacts/finetuned/` + `artifacts/finetuned-evaluation/`. The gallery requires the original dataset files. Each checkpoint hash is verified against its evaluation. Weights and data are excluded from Git, so a fresh checkout must run the reproduction commands below. If no model exists, it displays a setup message and no prediction. Train with the commands below. The UI accepts 8-bit JPEG/PNG exports, at most 10 MB and 20 megapixels; DICOM and high-bit-depth windowing are outside the current scope. Use de-identified research images. Uploads are processed in memory by the Python server and not written to disk by the app. A remotely hosted version would transmit images to that server.

## Reproduce the baseline

1. Download [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), a mirror of [Kermany, Zhang & Goldbaum's source dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2). The source is licensed CC BY 4.0; retain attribution when redistributing images. The archive includes duplicate directory trees: use just one folder directly containing `train`, `val`, and `test`, each with `NORMAL` and `PNEUMONIA`.
2. Place the extracted data under `data/chest_xray/` and run:

```bash
python -m pneumolens.prepare --root data/chest_xray --output data/manifest.csv
TORCH_HOME=artifacts/torch python -m pneumolens.train --manifest data/manifest.csv
python -m pneumolens.evaluate --manifest data/manifest.csv --checkpoint artifacts/baseline/model.pt
python -m pneumolens.examples
python serve.py
```

Training downloads official ImageNet weights on first use. Device selection prefers CUDA, then Apple MPS, then CPU; override with `--device cpu`. Training caches frozen backbone features in memory, then optimizes a two-class linear head with AdamW and training-only class weights. No augmentation or backbone fine-tuning is used in this baseline. Select the best checkpoint by unweighted validation cross-entropy, with a maximum of 40 epochs and patience of 8. The decision threshold is fixed at 0.5, not optimized on the test set. Weighted training loss and unweighted validation loss are different objectives and should not be directly compared.

Seeds fix the data split and initialization; numerical results can vary across hardware or dependency versions. The committed environment lock and training report record the baseline environment.

The preparation command preserves the supplied test split, excludes overlapping filename groups or identical decoded images from development, deduplicates development images, and creates a seeded 80/20 group-stratified train/validation split from the original development folders. The audit records removed images and final class counts. The manifest validator also rejects cross-split group or pixel-duplicate overlap before training or evaluation.

**Grouping limitation:** `personNN`, `IM-NNNN`, and `NORMAL2-IM-NNNN` are filename-derived groups, not independently verified patient records. This reduces obvious leakage but does not prove complete patient independence across naming families or detect all near-duplicates. Use verified patient metadata for stronger claims. The manifest format is `path,label,patient_id,split`; paths can be relative to the manifest. Neither absent overlap nor high test accuracy establishes clinical generalization.

The test command is separate from checkpoint selection and writes sensitivity, specificity, precision, F1, AUROC, average precision, Brier score, and a confusion matrix. `predictions.csv` includes every test result for reviewing false positives and false negatives. Average precision summarizes the precision-recall curve; it is not trapezoidal PR-AUC. Undefined metrics are `null`. Scores are uncalibrated softmax outputs. Do not repeatedly tune against test results.

The examples command exports the first test image of each outcome (true positive, true negative, false positive, false negative), plus real pneumonia-class Grad-CAM overlays. These examples are deliberately selected by outcome, not representative estimates or hand-picked attractive heatmaps. Copies retain source attribution in `docs/examples.json`; overlay images are modified versions of the source images. The app checks that displayed test metrics match the current checkpoint.

## Measured first baseline

One training run, seed 42, selected epoch 11. Development contains 3,800 training and 953 validation images. The supplied test split contains 624 images (234 NORMAL, 390 PNEUMONIA). The conservative split audit excluded 457 overlapping development images and 22 development duplicates.

| Metric | Validation | Held-out test |
| --- | ---: | ---: |
| Accuracy | 97.1% | 78.0% |
| Sensitivity / pneumonia recall | 97.4% | 99.0% |
| Specificity / normal recall | 96.3% | 43.2% |
| AUROC | 0.995 | 0.945 |
| Average precision | 0.998 | 0.960 |

The test confusion matrix is `[[101, 133], [4, 386]]`, with rows = actual and columns = predicted NORMAL/PNEUMONIA. At the fixed 0.5 threshold this baseline catches most pneumonia-labeled images but flags **133 of 234 normal-labeled images** as pneumonia. The validation-to-test gap is substantial. This is a working research baseline with a generalization problem, not a high-performing clinical classifier. The current evaluation does not determine the cause of that gap. Do not hide it by presenting validation accuracy as test accuracy or by tuning repeatedly against this test set.

Full records: [test metrics](test-metrics.json), [training history](training.json), [split audit](split-audit.json), and [model card](model-card.md). The interface includes correctly classified examples and both kinds of mistakes. Grad-CAM does not correct a wrong classification.

## How Grad-CAM works

The implementation is in `pneumolens/gradcam.py`, rather than hidden behind an explanation library:

```text
alpha[k] = mean over spatial positions of d(class_logit)/d(activation[k])
CAM = ReLU(sum over channels of alpha[k] * activation[k])
```

Capture the last residual block's activations, differentiate the requested class logit, combine channels using average gradients, and normalize positive activation to [0, 1]. The native 7 × 7 map is resized to the 224 × 224 model input; padding is removed before resizing to the original image. This keeps the overlay geometrically aligned without cutting off the field of view. A zero map remains zero and displays an explicit message. Each map is normalized independently, so intensity is not comparable across scans or classes.

The same deterministic preprocessing is used in training, evaluation, and the app: RGB conversion, aspect-ratio-preserving resize and black padding to 224 × 224, then ImageNet channel normalization. RGB replicates grayscale channels; it does not create additional medical information. This full-field transform deliberately differs from ImageNet's standard center crop. Padding and other artifacts can still influence the model.

## Custom explorer and measured follow-up

The redesigned explorer includes Original / Overlay / Compare modes, a draggable divider, shared zoom and pan, opacity control, class selection and PNG export. All 624 test cases are available through an outcome-filtered, paginated gallery. Confusion-matrix cells lead to their cases. Both model versions remain inspectable.

Uploads are processed in memory; the browser keeps at most eight recent results until reload. Displayed source images have a maximum dimension of 1280 pixels; exported PNGs add a context footer, and CAMs are transmitted as aligned 8-bit grayscale maps before coloring in the browser. Opacity changes do not rerun inference. The server binds to loopback, checks Host/Origin, and exposes fixed routes. It is a local demonstration without authentication or production hosting infrastructure.

The [fine-tuning protocol](finetuning-protocol.md) was recorded before training. One run trained `layer4` and the head, kept BatchNorm running statistics fixed, and selected epoch three by validation cross-entropy after six epochs. The baseline is preserved. Threshold remains 0.50. The test was already examined for the baseline, so this comparison is **exploratory test reuse**, not independent validation.

```bash
python -m pneumolens.finetune --manifest data/manifest.csv
python -m pneumolens.evaluate --manifest data/manifest.csv --checkpoint artifacts/finetuned/model.pt --output artifacts/finetuned-evaluation --test-reused
python -m pneumolens.sanity
```

| Test metric | Frozen baseline | Fine-tuned |
| --- | ---: | ---: |
| Accuracy | 78.0% | 78.8% |
| Pneumonia recall | 99.0% | 99.7% |
| Normal recall | 43.2% | 44.0% |
| AUROC | 0.945 | 0.955 |
| Average precision | 0.960 | 0.964 |
| Brier score (lower is better) | 0.167 | 0.179 |
| False positives | 133 | 131 |
| False negatives | 4 | 1 |

Fine-tuning slightly improves classifications but still falsely flags **131 of 234 normal-labeled images**. Probability error measured by Brier score worsens. Fine-tuned confusion matrix: `[[103,131],[1,389]]`. These results support an error-analysis portfolio project, not clinical readiness. Records: [fine-tuned metrics](finetuned-test-metrics.json), [training history](finetuned-training.json), and [verification](verification.md).

## Explanation sanity check

The recorded experiment selects the first two validation filenames per class and compares pneumonia-class Grad-CAM before and after randomizing the head, then the head plus `layer4`. Earlier layers remain trained. The [JSON report](sanity/results.json) includes the seed, checkpoint/manifest hashes, per-image Pearson correlations, absolute map changes, and descriptive border-activation measures. The [comparison sheet](sanity/randomization.png) shows all four cases.

Maps change after randomization; some randomized overlays still appear visually plausible. Two normal-labeled examples place 86.2% and 71.6% of trained pneumonia-class activation mass in the outer 10% along each edge (approximately 36% of image area). One map includes the region near a marker. These observations motivate investigation but do not prove a causal shortcut. Separately normalized maps can look prominent even when the pneumonia score is very low.

This small descriptive check draws on [Adebayo et al., Sanity Checks for Saliency Maps](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html). It is not a full replication, a pass/fail benchmark, or localization validation. No random-label training, full-network randomization, expert masks, or medical correctness assessment is included.

## Portfolio story

| Project | Main evidence of skill |
| --- | --- |
| Neural Canvas | Forward/backpropagation, optimization, decision boundaries |
| Chroma Lab | K-means++, perceptual color, practical image processing |
| PneumoLens | Transfer learning, class imbalance, reproducible evaluation, gradient-based interpretation |

Suggested description: **An interactive chest X-ray classification explorer using PyTorch and Grad-CAM, with reproducible data splits and transparent evaluation of model limitations.**

Useful next experiments: report variability over several development seeds; study false positives, false negatives, borders, and text artifacts using development data; evaluate a separate source population. A last-stage fine-tuning run and a small parameter-randomization check are now recorded below. Localization claims require suitable expert annotations and a localization evaluation protocol.

## Tests and source map

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
node --test tests/viewer.test.mjs
node --input-type=module --check < web/app.js
```

Tests verify Grad-CAM against an analytically known spatial map, class dependence, zero activation, hook cleanup, frozen-backbone gradients, padding inversion, known confusion metrics, duplicate leakage, and unsupported image depth. These checks establish implementation behavior, not clinical effectiveness. `requirements-lock.txt` records the development environment; `requirements.txt` allows compatible installations on other platforms.

```text
serve.py                Custom local explorer on port 8502
web/                    HTML/CSS, UI state, canvas viewer
app.py                  Original Streamlit explorer on port 8501
pneumolens/server.py    Local API and actual model inference
pneumolens/finetune.py  Validation-selected last-stage fine-tuning
pneumolens/sanity.py    Parameter-randomization experiment
pneumolens/model.py     Architecture and checkpoint validation
pneumolens/gradcam.py   Explicit Grad-CAM implementation
pneumolens/imaging.py   Shared preprocessing and overlay alignment
pneumolens/prepare.py   Dataset manifest and split audit
pneumolens/data.py      Split validation and image loading
pneumolens/train.py     Frozen-feature baseline and model selection
pneumolens/evaluate.py  Test metrics and per-image predictions
tests/                  Mathematical and data-integrity checks
docs/                   Model card and measured run records
```

## References

- [Selvaraju et al. — Grad-CAM](https://arxiv.org/abs/1610.02391)
- [Kermany, Zhang & Goldbaum — original dataset, version 2](https://data.mendeley.com/datasets/rscbjbr9sj/2)
- [Torchvision — ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

Implementation created with AI assistance. Review and understand the model, data assumptions, and error analysis before presenting the project as your work.
