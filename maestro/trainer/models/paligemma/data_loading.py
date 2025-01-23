import logging
import os
from functools import partial
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from maestro.trainer.common.data_loaders.datasets import DetectionDataset


def prepare_data_loaders(
    dataset_location: str,
    train_batch_size: int,
    processor: AutoProcessor,
    device: torch.device,
    num_workers: int = 0,
    test_batch_size: Optional[int] = None,
    test_loaders_workers: Optional[int] = None,
) -> tuple[
    DataLoader,
    Optional[DataLoader],
    Optional[DataLoader],
]:
    test_batch_size = test_batch_size or train_batch_size
    test_loaders_workers = test_loaders_workers or num_workers
    train_data_loader = prepare_detection_data_loader(
        dataset_location=dataset_location,
        split_name="train",
        batch_size=train_batch_size,
        processor=processor,
        device=device,
        num_workers=num_workers,
        shuffle=True,
    )
    if train_data_loader is None:
        raise RuntimeError("Could not initialise train data loader")
    valid_data_loader = prepare_detection_data_loader(
        dataset_location=dataset_location,
        split_name="valid",
        batch_size=test_batch_size,
        processor=processor,
        device=device,
        num_workers=test_loaders_workers,
        shuffle=False,
    )
    test_data_loader = prepare_detection_data_loader(
        dataset_location=dataset_location,
        split_name="test",
        batch_size=test_batch_size,
        processor=processor,
        device=device,
        num_workers=test_loaders_workers,
        shuffle=False,
    )
    return train_data_loader, valid_data_loader, test_data_loader


def prepare_detection_data_loader(
    dataset_location: str,
    split_name: str,
    batch_size: int,
    processor: AutoProcessor,
    device: torch.device,
    num_workers: int = 0,
    shuffle: bool = True,
) -> Optional[DataLoader]:
    dataset = prepare_detection_dataset(
        dataset_location=dataset_location,
        split_name=split_name,
    )
    if dataset is None:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=device),
        num_workers=num_workers,
        shuffle=shuffle,
    )


def prepare_detection_dataset(
    dataset_location: str,
    split_name: str,
) -> Optional[DetectionDataset]:
    image_directory_path = os.path.join(dataset_location, split_name)
    jsonl_file_path = os.path.join(dataset_location, split_name, "annotations.jsonl")
    if not os.path.exists(image_directory_path):
        logging.warning(f"Could not data directory: {image_directory_path}")
        return None
    if not os.path.exists(jsonl_file_path):
        logging.warning(f"Could not find JSONL file: {jsonl_file_path}")
        return None
    return DetectionDataset(
        jsonl_file_path=jsonl_file_path,
        image_directory_path=image_directory_path,
    )


def collate_fn(
    batch: tuple[list[str], list[str], list[Image.Image]],
    processor: AutoProcessor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    questions, answers, images = zip(*batch)
    # TO DO : Need to check i
    inputs = processor(
        text=list(questions), images=list(images), suffix=list(answers), return_tensors="pt", padding="longest"
    ).to(device)
    return inputs
