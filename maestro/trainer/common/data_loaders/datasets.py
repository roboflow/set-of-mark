import json
import logging
import os
from typing import Any, Callable, Optional

from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROBOFLOW_JSONL_FILENAME = "annotations.jsonl"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class RoboflowJSONLDataset(Dataset):
    """
    Dataset for loading images and annotations from a Roboflow JSONL dataset.

    Args:
        jsonl_file_path (str): Path to the JSONL file containing dataset entries.
        image_directory_path (str): Path to the directory containing images.
    """

    def __init__(self, jsonl_file_path: str, image_directory_path: str) -> None:
        if not os.path.exists(jsonl_file_path):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_file_path}")
        if not os.path.isdir(image_directory_path):
            raise NotADirectoryError(f"Image directory not found: {image_directory_path}")

        self.image_directory_path = image_directory_path
        self.entries = self._load_entries(jsonl_file_path)

    @staticmethod
    def _load_entries(jsonl_file_path: str) -> list[dict]:
        with open(jsonl_file_path) as file:
            try:
                return [json.loads(line) for line in file]
            except json.JSONDecodeError:
                logging.exception("Error parsing JSONL file.")
                raise

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[Image.Image, dict]:
        if idx >= len(self.entries):
            raise IndexError("Index out of range")

        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry["image"])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
        except OSError as e:
            raise OSError(f"Error opening image file {image_path}: {e}")

        return image, entry


def load_split_dataset(dataset_location: str, split_name: str) -> Optional[Dataset]:
    """
    Load a dataset split from the specified location.

    Args:
        dataset_location (str): Path to the dataset directory.
        split_name (str): Name of the dataset split (e.g., "train", "valid", "test").

    Returns:
        Optional[Dataset]: A dataset object for the split, or `None` if the split does not exist.
    """
    jsonl_file_path = os.path.join(dataset_location, split_name, ROBOFLOW_JSONL_FILENAME)
    image_directory_path = os.path.join(dataset_location, split_name)

    if not os.path.exists(jsonl_file_path) or not os.path.exists(image_directory_path):
        logging.warning(f"Dataset split {split_name} not found at {dataset_location}")
        return None

    return RoboflowJSONLDataset(jsonl_file_path, image_directory_path)


def create_data_loaders(
    dataset_location: str,
    train_batch_size: int,
    train_collect_fn: Callable[[list[Any]], Any],
    train_num_workers: int = 0,
    test_batch_size: Optional[int] = None,
    test_collect_fn: Optional[Callable[[list[Any]], Any]] = None,
    test_num_workers: Optional[int] = None,
) -> tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create DataLoader instances for training, validation, and testing datasets.

    Args:
        dataset_location (str): Path to the dataset directory.
        train_batch_size (int): Batch size for the training dataset. Must be a positive integer.
        train_collect_fn (Callable[[List[Any]], Any]): Function to collate training samples into a batch.
        train_num_workers (int): Number of worker threads for the training DataLoader. Defaults to 0.
        test_batch_size (Optional[int]): Batch size for validation and test datasets. Defaults to the value of
            `train_batch_size` if not provided.
        test_collect_fn (Optional[Callable[[List[Any]], Any]]): Function to collate validation and test samples into a
            batch. Defaults to the value of `train_collect_fn` if not provided.
        test_num_workers (Optional[int]): Number of worker threads for validation and test DataLoaders. Defaults to the
            value of `train_num_workers` if not provided.

    Returns:
        Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]: A tuple containing the DataLoader for the
            training dataset, and optionally for the validation and testing datasets. If a dataset split is not found,
            the corresponding DataLoader is `None`.

    Raises:
        ValueError: If batch sizes are not positive integers or no dataset splits are found.
    """
    if train_batch_size <= 0:
        raise ValueError("train_batch_size must be a positive integer.")

    test_batch_size = test_batch_size or train_batch_size
    if test_batch_size <= 0:
        raise ValueError("test_batch_size must be a positive integer.")

    test_num_workers = test_num_workers or train_num_workers
    test_collect_fn = test_collect_fn or train_collect_fn

    train_dataset = load_split_dataset(dataset_location, "train")
    valid_dataset = load_split_dataset(dataset_location, "valid")
    test_dataset = load_split_dataset(dataset_location, "test")

    if not any([train_dataset, valid_dataset, test_dataset]):
        raise ValueError("No dataset splits found. Ensure the dataset is correctly structured.")

    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=train_num_workers,
            collate_fn=train_collect_fn,
        )
        if train_dataset
        else None
    )

    valid_loader = (
        DataLoader(
            valid_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=test_num_workers,
            collate_fn=test_collect_fn,
        )
        if valid_dataset
        else None
    )

    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=test_num_workers,
            collate_fn=test_collect_fn,
        )
        if test_dataset
        else None
    )

    return train_loader, valid_loader, test_loader
