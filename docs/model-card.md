# PneumoLens model card

**Intended use:** educational computer vision and model-interpretation portfolio demonstration.

**Task:** predict the dataset classes NORMAL and PNEUMONIA from an 8-bit chest X-ray export. NORMAL is a dataset label, not a finding that a patient has no disease. No differential diagnosis, pneumonia subtype prediction, severity grading, segmentation, or triage recommendation is provided.

**Architecture:** ImageNet-pretrained ResNet-18 with a trained two-class linear head; backbone frozen. Grad-CAM targets the last residual block's output and a selected pre-softmax logit.

**Dataset provenance:** Kermany, Zhang & Goldbaum, *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*, version 2, DOI 10.17632/rscbjbr9sj.2; downloaded via the linked Kaggle chest X-ray mirror. Only chest X-ray folders are used. The source lists CC BY 4.0. No private patient data is used.

**Development split:** supplied train/val pooled, exact duplicates removed, conservatively grouped by filenames, then split 80/20 by group with class stratification and seed 42. Test remains the supplied test partition. Any development group or exact pixel duplicate overlapping test is excluded. These checks do not verify identities across filename naming families or detect all near-duplicates.

**Selection and evaluation:** best validation cross-entropy selects the checkpoint. Class weights use only training counts. Threshold 0.5 is fixed before test evaluation. The test is evaluated separately and per-image outcomes are exported. One source dataset cannot demonstrate external validity. Multiple images per patient/group may make image-level measurements optimistic about independent-case performance.

**Interpretation limits:** Grad-CAM is a coarse model-attribution map. Red/amber indicates stronger positive activation after per-image normalization, not a probability or tissue finding. No localization ground truth is used. High attention on image markers, borders, or padding may reveal shortcuts. A plausible-looking heatmap does not prove that the classifier learned valid medical features.

**Deployment:** local Python HTTP server with a custom browser/canvas viewer; original Streamlit demo retained, no authentication or clinical workflow integration. Images are handled in memory and not saved by app code. The model cannot validate image modality or detect distribution shift; arbitrary images can still receive scores. DICOM and high-bit-depth processing are unsupported. No clinical validation, expert reader study, subgroup fairness analysis, or external evaluation has been performed.

**Measured results:** see `docs/test-metrics.json` and `docs/training.json` after a completed run. Report the exact split audit with these results; do not describe these scores as clinical accuracy or diagnostic confidence.

First completed run: validation accuracy 97.1%; test accuracy 78.0%, sensitivity 99.0%, specificity 43.2%, AUROC 0.945. Test outcomes: 101 true negatives, 133 false positives, 4 false negatives, 386 true positives. The large validation-to-test gap and high false-positive rate are material limitations. Its cause is not established by this experiment. The baseline artifact remains unchanged; a separately saved follow-up is described below.

## Follow-up model and explanation experiment

The [fine-tuning protocol](finetuning-protocol.md) was recorded before one run. It trains the last residual stage and head while holding earlier layers and BatchNorm running statistics fixed. Epoch three was selected by validation cross-entropy after six epochs. The fixed threshold remains 0.50. Test accuracy is 78.8%, sensitivity 99.7%, specificity 44.0%, AUROC 0.955; outcomes are 103 true negatives, 131 false positives, one false negative and 389 true positives. Brier score worsens from 0.167 to 0.179.

This is one exploratory comparison on the test set already examined for the baseline. No further tuning was performed against these results. Fine-tuning does not resolve the generalization problem. Full records: [metrics](finetuned-test-metrics.json) and [training](finetuned-training.json).

The [parameter check](sanity/results.json) uses four validation images selected by filename, not outcome. Randomizing head and last-stage parameters changes maps, but some randomized overlays look plausible and some trained activation lies near borders or markers. All maps explain PNEUMONIA, including normal-labeled examples. One seed, descriptive statistics, no localization annotations, no causal artifact test and no pass/fail claim.

The custom app caches at most eight recent results in browser memory until reload. Displayed source images are at most 1280 pixels on the longest side; exports add a context footer; aligned CAMs are transmitted at 8-bit precision. It remains a local research demo, not a clinical system.
