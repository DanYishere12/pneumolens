# Presenting PneumoLens

## Project description

An interactive chest X-ray classifier built with PyTorch and Grad-CAM. It combines transfer learning, reproducible evaluation, synchronized image comparisons, and an inspectable gallery of correct and incorrect predictions.

## A 90-second walkthrough

1. **0–15 seconds:** Open the explorer. Explain that it predicts dataset labels and shows positive model influence. If using the recorded demo, say that these are saved outputs from real checkpoints; new-image inference runs in the full app.
2. **15–35 seconds:** Drag the divider, zoom, and adjust opacity. Explain that both views share the same transform so image regions stay aligned. Switch the explained class: the heatmap changes while the prediction stays fixed.
3. **35–55 seconds:** Open Evaluation. State the fixed threshold and reused-test limitation. Show the 131 false positives and one false negative for the fine-tuned model. Click an error category and inspect an actual case.
4. **55–75 seconds:** Compare the models. Accuracy improves slightly, but normal recall stays low and Brier score worsens. Explain why reporting accuracy alone would conceal a major weakness.
5. **75–90 seconds:** Show the parameter-randomization comparison in the documentation or full app. Some randomized maps look plausible. Finish with the next experiment: independent-source evaluation and development-set investigation of image artifacts.

## Interview discussion points

- **Why a baseline first?** Frozen ImageNet features with a linear head provide a simple reference for measuring a more expensive fine-tuning experiment.
- **How was the follow-up chosen?** Unfreeze `layer4` and the head, retain BatchNorm statistics, and select by validation cross-entropy. The threshold stays at 0.50. Do not claim test independence after its reuse.
- **What does Grad-CAM explain?** It weights final-block activations by spatially averaged gradients of a selected pre-softmax logit, then retains positive contributions. The native map is coarse (7 × 7).
- **Why preserve aspect ratio?** Letterboxing avoids cutting off the field of view. The same preprocessing is used throughout; padding is removed before mapping attribution back to the displayed image.
- **How is leakage reduced?** Exclude development filename groups and exact decoded-image duplicates that overlap the supplied test set. Filename groups are a proxy, not verified patient metadata.
- **What failed?** The substantial validation/test gap and high false-positive rate remain unresolved. Per-image normalized heatmaps can also emphasize weak activation or border regions.
- **What would improve the evidence?** Independent-source data, verified patient identities, repeated development seeds, artifact analysis, and appropriate expert annotations before making localization claims.

## Suggested résumé bullet

Built a PyTorch chest X-ray classification explorer with an explicit Grad-CAM implementation, synchronized canvas visualization, reproducible grouped data preparation, and transparent comparison of baseline and fine-tuned models across 624 test images.

Use this only once you can explain the implementation and your use of AI assistance. Avoid claiming a diagnostic tool, validated lesion localization, or 98% test accuracy; the higher accuracy belongs to development validation.
