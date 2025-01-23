import os
from typing import Optional

import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from maestro.trainer.common.configuration.env import CUDA_DEVICE_ENV, DEFAULT_CUDA_DEVICE

DEFAULT_PALIGEMMA_MODEL_ID = "google/paligemma-3b-pt-224"
DEFAULT_PALIGEMMA_MODEL_REVISION = "float16"
DEVICE = torch.device("cpu") if not torch.cuda.is_available() else os.getenv(CUDA_DEVICE_ENV, DEFAULT_CUDA_DEVICE)


class CheckpointManager:
    """Manages checkpoints for model training.

    This class handles saving and retrieving model checkpoints during training.

    Attributes:
        training_dir (str): Directory where checkpoints will be saved.
        best_val_loss (float): Best validation loss achieved so far.
        latest_checkpoint_dir (str): Directory for the latest checkpoint.
        best_checkpoint_dir (str): Directory for the best checkpoint.
    """

    def __init__(self, training_dir: str):
        """Initializes the CheckpointManager.

        Args:
            training_dir (str): Directory where checkpoints will be saved.
        """
        self.training_dir = training_dir
        self.best_val_loss = float("inf")
        self.latest_checkpoint_dir = os.path.join(training_dir, "checkpoints", "latest")
        self.best_checkpoint_dir = os.path.join(training_dir, "checkpoints", "best")

    def save_latest(self, processor: AutoProcessor, model: PaliGemmaForConditionalGeneration):
        """Saves the latest model checkpoint.

        Args:
            processor (AutoProcessor): The processor to save.
            model (PaliGemmaForConditionalGeneration): The model to save.
        """
        save_model(self.latest_checkpoint_dir, processor, model)

    def save_best(self, processor: AutoProcessor, model: PaliGemmaForConditionalGeneration, val_loss: float):
        """Saves the best model checkpoint if the validation loss improves.

        Args:
            processor (AutoProcessor): The processor to save.
            model (PaliGemmaForConditionalGeneration): The model to save.
            val_loss (float): The current validation loss.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            save_model(self.best_checkpoint_dir, processor, model)
            print(f"New best model saved with validation loss: {self.best_val_loss}")

    def get_best_model_path(self):
        """Returns the path to the best model checkpoint.

        Returns:
            str: Path to the best model checkpoint.
        """
        return self.best_checkpoint_dir


def save_model(
    target_dir: str,
    processor: AutoProcessor,
    model: PaliGemmaForConditionalGeneration,
) -> None:
    """Saves the model and processor to the specified directory.

    Args:
        target_dir (str): Directory where the model and processor will be saved.
        processor (AutoProcessor): The processor to save.
        model (PaliGemmaForConditionalGeneration): The model to save.
    """
    os.makedirs(target_dir, exist_ok=True)
    processor.save_pretrained(target_dir)
    model.save_pretrained(target_dir)


def load_model(
    model_id_or_path: str = DEFAULT_PALIGEMMA_MODEL_ID,
    revision: str = DEFAULT_PALIGEMMA_MODEL_REVISION,
    device: torch.device = DEVICE,
    cache_dir: Optional[str] = None,
) -> tuple[AutoProcessor, PaliGemmaForConditionalGeneration]:
    """Loads a PaliGemma model and its associated processor.

    Args:
        model_id_or_path: The identifier or path of the model to load.
        revision: The specific model revision to use.
        device: The device to load the model onto.
        cache_dir: Directory to cache the downloaded model files.

    Returns:
        A tuple containing the loaded processor and model.

    Raises:
        ValueError: If the model or processor cannot be loaded.
    """
    processor = AutoProcessor.from_pretrained(
        model_id_or_path,
        trust_remote_code=True,
        revision=revision,
    )
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id_or_path,
        trust_remote_code=True,
        revision=revision,
        cache_dir=cache_dir,
    ).to(device)
    return processor, model
