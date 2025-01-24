from typing import Any

from PIL import Image
from transformers import PaliGemmaProcessor

MAX_LENGTH = 512


def train_collate_fn(batch: list[tuple[Image.Image, dict[str, Any]]], processor: PaliGemmaProcessor):
    images, data = zip(*batch)
    prefixes = ["<image>" + entry["prefix"] for entry in data]
    suffixes = [entry["suffix"] for entry in data]

    inputs = processor(
        text=prefixes,
        images=images,
        return_tensors="pt",
        suffix=suffixes,
        padding=True,
        truncation="only_second",
        max_length=MAX_LENGTH,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    pixel_values = inputs["pixel_values"]
    labels = inputs["labels"]

    return input_ids, attention_mask, token_type_ids, pixel_values, labels


def evaluation_collate_fn(batch: list[tuple[Image.Image, dict[str, Any]]], processor: PaliGemmaProcessor):
    images, data = zip(*batch)
    prefixes = ["<image>" + entry["prefix"] for entry in data]
    suffixes = [entry["suffix"] for entry in data]

    inputs = processor(text=prefixes, images=images, return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    return input_ids, attention_mask, pixel_values, prefixes, suffixes
