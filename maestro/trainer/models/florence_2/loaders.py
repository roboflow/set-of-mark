from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor


def collate_fn(
    batch: list[tuple[Image.Image, dict[str, Any]]],
    processor: AutoProcessor,
    device: torch.device,
):
    images, data = zip(*batch)
    prefixes = [entry["prefix"] for entry in data]
    suffixes = [entry["suffix"] for entry in data]
    inputs = processor(text=list(prefixes), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, prefixes, suffixes, images
