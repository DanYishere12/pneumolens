import torch
from torch.nn import functional as F


def grad_cam(model, target_layer, image, target_class):
    """Gradient-weighted CAM for one image, targeting a pre-softmax class logit.

    Use autograd.grad on captured activations, avoiding backward hooks and
    parameter-gradient accumulation. The caller owns this model instance.
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Grad-CAM expects one image with shape (1, C, H, W).")
    captured = []
    handle = target_layer.register_forward_hook(lambda module, inputs, output: captured.append(output))
    was_training = model.training
    try:
        model.eval()
        with torch.enable_grad():
            logits = model(image.detach().clone().requires_grad_(True))
            if not 0 <= target_class < logits.shape[1]:
                raise ValueError("Target class is out of range.")
            activation = captured[-1]
            gradient, = torch.autograd.grad(logits[0, target_class], activation)
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activation).sum(dim=1, keepdim=True).relu()
            cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
            peak = cam.amax()
            cam = cam / peak.clamp_min(1e-12)
        return logits.detach().softmax(dim=1)[0].cpu().numpy(), cam[0, 0].detach().cpu().numpy()
    finally:
        handle.remove()
        model.train(was_training)
