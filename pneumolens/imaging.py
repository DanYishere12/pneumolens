import io

import numpy as np
import torch
from PIL import Image, ImageOps

SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def read_image(source):
    with Image.open(source) as image:
        if image.format not in {"JPEG", "PNG"}:
            raise ValueError("Use a JPEG or PNG image.")
        if image.width * image.height > 20_000_000:
            raise ValueError("Image exceeds the 20-megapixel limit.")
        if image.mode in {"I", "F"} or image.mode.startswith("I;16"):
            raise ValueError("Use an 8-bit display export; high-bit-depth scans require windowing.")
        return ImageOps.exif_transpose(image).convert("RGB")


def preprocess(image):
    """Preserve field of view and aspect ratio; return padding geometry for CAM alignment."""
    image = image.convert("RGB")
    scale = min(SIZE / image.width, SIZE / image.height)
    width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    left, top = (SIZE - width) // 2, (SIZE - height) // 2
    canvas = Image.new("RGB", (SIZE, SIZE))
    canvas.paste(image.resize((width, height), Image.Resampling.BILINEAR), (left, top))
    tensor = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 255
    return (tensor - MEAN) / STD, (left, top, width, height)


def unpad_cam(cam, geometry, original_size):
    left, top, width, height = geometry
    field = Image.fromarray(cam.astype(np.float32))
    field = field.crop((left, top, left + width, top + height))
    return np.clip(np.asarray(field.resize(original_size, Image.Resampling.BILINEAR)), 0, 1)


def overlay(image, cam, opacity=0.55):
    if not 0 <= opacity <= 1:
        raise ValueError("Opacity must be between zero and one.")
    # Transparent low activation, amber-to-red high activation.
    color = np.stack([np.ones_like(cam), 0.85 * (1 - cam), 0.12 * (1 - cam)], axis=-1)
    alpha = (cam * opacity)[..., None]
    result = np.asarray(image.convert("RGB"), dtype=np.float32) * (1 - alpha) + color * 255 * alpha
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
