# Experiment 02: last-block fine-tuning

Protocol written before this experiment is run. The baseline's test results have already been viewed; this is an exploratory follow-up, not a new independent validation.

- Initialize from the saved baseline checkpoint selected at epoch 11.
- Preserve the existing manifest, image preprocessing, seed 42, training class weights, and threshold 0.5.
- Train only ResNet layer4 and the classification head. Keep all BatchNorm running statistics fixed, including those in layer4. No augmentation in this controlled first comparison.
- AdamW: layer4 learning rate 0.00001, head learning rate 0.0001, weight decay 0.001; batch size 32.
- At most 8 epochs, patience 3; select by unweighted validation cross-entropy. Treat the baseline validation loss as the initial incumbent, so an unsuccessful run cannot silently replace it.
- Save separate checkpoints and records under artifacts/finetuned. Never overwrite the baseline.
- Evaluate the selected candidate on the existing test partition once for an explicitly retrospective comparison. Do not change settings in response to that test result. An independent external evaluation remains outstanding.
- Report all epochs, selected epoch, sensitivity, specificity, accuracy, AUROC, average precision, and both error types. No promise of improvement is made.

Implementation follows [PyTorch's fine-tuning approach](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). This experiment tests a particular configuration, not all fine-tuning strategies.
