# Verification record

## Current checks

- **16 Python tests pass.** Grad-CAM math, class dependence, zero maps, hook cleanup, padding inversion, split integrity, model/report consistency, stale prediction detection, API boundaries, uploads, and fixed BatchNorm behavior are covered.
- **Six JavaScript tests pass.** Viewer blending/placement plus the recorded demo's actual checksums, checkpoint provenance, image/CAM dimensions, class consistency, gallery versus full-evaluation counts, and upload rejection are covered.
- Frontend module syntax check passes. CI runs Python 3.11 and Node 22.
- The recorded package contains five selected real test scans, both evaluated models and both explained classes. Assets occupy approximately 4.3 MB. The exporter verifies scores against the saved prediction CSVs.

The Python run took 467 seconds because of unusually slow initial dependency-file reads in this local environment. The test suite completed successfully; package versions and model weights were not changed to work around it.

## Light-and-teal UI pass — September 17, 2026

- Replaced the dark page theme with an off-white background, white cards, darker text, and teal accents. The X-ray viewer retains its own dark palette.
- Enlarged supporting text, labels, form inputs, and viewer controls. The prediction panel has more room; phone layouts keep readable text sizes and stack the controls and method cards.
- Visually reviewed the full inference app at rendered widths of 1280, 768, and 390 pixels, and all three recorded-demo pages at 354 pixels. No horizontal page overflow was observed at these widths.
- Checked confusion-matrix navigation to the false-negative case, keyboard comparison adjustment (50% to 51%), zoom (100% to 125%), explained-class switching, and zero-opacity rendering. The false-negative prediction remained Normal with a 9.6% pneumonia score after changing the explained class.
- Representative text/background pairs exceed 4.5:1 contrast: body copy, supporting text, teal controls and matrix cells, error matrix cells, and dark-viewer supporting text. This is a palette check, not a complete accessibility audit.
- The full app reported no browser console warnings or errors. Temporary viewport overrides were reset.
- Changes are limited to CSS and documentation. The existing Python and JavaScript test results above are from the earlier implementation pass; those suites were not rerun for this visual change. Model weights, predictions, heatmap colors, and inference code were not changed.

## Browser verification

- The standard-library demo runs on port 8503 without installing ML packages. Its banner explicitly identifies saved outputs and explains that no inference runs in this mode. Upload controls are absent.
- The gallery shows five recorded examples while Evaluation reports 624 test images. Clicking the fine-tuned false-negative matrix cell selects `person154_bacteria_728.jpeg`, with the saved pneumonia score of 9.6%.
- Switching that example's explained class to NORMAL preserves its Normal prediction and 9.6% pneumonia score.
- Phone layouts at 390 pixels were visually reviewed with actual images and evaluation data. Document width was 390 pixels, with no horizontal overflow. The upload button remains readable, controls wrap and the inspector stacks vertically.
- The full inference app was inspected at 1280 pixels. Pointer dragging moved the comparison slider from 50% to about 68%; zoom reached 150%, and panning moved the displayed image. Original and overlay retain shared geometry.
- Exporting while zoomed saved a valid 1106 × 923 PNG: the full 1106 × 762 source image plus an annotation footer. The downloaded file was decoded and visually inspected. It includes model, explained class, opacity, uncalibrated score and image attribution. See [actual export](export-example.png).
- The full app was reset and reloaded successfully; no browser console errors were reported. Temporary viewport overrides were reset and the temporary demo test tab was closed.

The in-app browser's download event did not fire, but the actual PNG was present in Downloads and was verified directly. This was a test-tool limitation, not a failed export.

## Earlier inference checks retained

A public example was uploaded through the file chooser. The fine-tuned model returned a pneumonia score of 90.9%, marked it as an unlabeled upload and displayed no dataset label. The filename did not determine the prediction. Earlier checks also covered baseline switching and opacity at zero.

These checks establish implementation behavior. They do not establish diagnostic effectiveness, explanation faithfulness, or medical localization. No model was retrained or retuned during this portfolio packaging pass.
