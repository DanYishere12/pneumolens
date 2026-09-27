from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

CLASSES = ["NORMAL", "PNEUMONIA"]
PREPROCESS = "rgb-letterbox224-imagenet-v1"


def build_model(pretrained=False):
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def load_checkpoint(path: str | Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("format_version") != 1
            or checkpoint.get("architecture") != "resnet18"
            or checkpoint.get("classes") != CLASSES
            or checkpoint.get("preprocess") != PREPROCESS
            or checkpoint.get("trained_epochs", 0) < 1):
        raise ValueError("Unsupported checkpoint. Train a model with this project's training command.")
    threshold = checkpoint.get("threshold")
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ValueError("Checkpoint has an invalid decision threshold.")
    model = build_model()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), checkpoint
