import os
from dataclasses import dataclass, field, replace
from typing import Literal, Optional, Union

import torch
from peft import PeftModel
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, get_scheduler

from maestro.trainer.common.peft import LoraInitLiteral, prepare_peft_model
from maestro.trainer.common.utils.file_system import create_new_run_directory
from maestro.trainer.common.utils.metrics import (
    BaseMetric,
    MeanAveragePrecisionMetric,
    MetricsTracker,
    display_results,
    save_metric_plots,
)
from maestro.trainer.common.utils.reproducibility import make_it_reproducible
from maestro.trainer.models.paligemma.checkpoints import (
    DEFAULT_PALIGEMMA_MODEL_ID,
    DEFAULT_PALIGEMMA_MODEL_REVISION,
    DEVICE,
    CheckpointManager,
    load_model,
)
from maestro.trainer.models.paligemma.data_loading import prepare_data_loaders
from maestro.trainer.models.paligemma.metrics import (
    extract_unique_detection_dataset_classes,
    postprocess_paligemma_output_for_mean_average_precision,
    run_predictions,
)


@dataclass(frozen=True)
class Configuration:
    """Configuration for training a PaliGemma model.

    This class encapsulates all the parameters needed for training a PaliGemma model,
    including dataset paths, model specifications, training hyperparameters, and output
    settings.

    Attributes:
        dataset (str): Path to the dataset used for training.
        model_id (str): Identifier for the PaliGemma model.
        revision (str): Revision of the model to use.
        device (torch.device): Device to use for training.
        cache_dir (Optional[str]): Directory to cache the model.
        epochs (int): Number of training epochs.
        optimizer (Literal["sgd", "adamw", "adam"]): Optimizer to use for training.
        lr (float): Learning rate for the optimizer.
        lr_scheduler (Literal["linear", "cosine", "polynomial"]): Learning rate
            scheduler.
        batch_size (int): Batch size for training.
        val_batch_size (Optional[int]): Batch size for validation.
        num_workers (int): Number of workers for data loading.
        val_num_workers (Optional[int]): Number of workers for validation data loading.
        lora_r (int): Rank of the LoRA update matrices.
        lora_alpha (int): Scaling factor for the LoRA update.
        lora_dropout (float): Dropout probability for LoRA layers.
        bias (Literal["none", "all", "lora_only"]): Which bias to train.
        use_rslora (bool): Whether to use RSLoRA.
        init_lora_weights (Union[bool, LoraInitLiteral]): How to initialize LoRA
            weights.
        output_dir (str): Directory to save output files.
        metrics (list[BaseMetric]): list of metrics to track during training.
    """

    dataset: str
    model_id: str = DEFAULT_PALIGEMMA_MODEL_ID
    revision: str = DEFAULT_PALIGEMMA_MODEL_REVISION
    device: torch.device = DEVICE
    cache_dir: Optional[str] = None
    epochs: int = 10
    optimizer: Literal["sgd", "adamw", "adam"] = "adamw"
    lr: float = 1e-5
    lr_scheduler: Literal["linear", "cosine", "polynomial"] = "linear"
    batch_size: int = 4
    val_batch_size: Optional[int] = None
    num_workers: int = 0
    val_num_workers: Optional[int] = None
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = True
    init_lora_weights: Union[bool, LoraInitLiteral] = "gaussian"
    output_dir: str = "./training/florence-2"
    metrics: list[BaseMetric] = field(default_factory=list)


def train(config: Configuration) -> None:
    make_it_reproducible(avoid_non_deterministic_algorithms=False)
    run_dir = create_new_run_directory(
        base_output_dir=config.output_dir,
    )
    config = replace(
        config,
        output_dir=run_dir,
    )
    checkpoint_manager = CheckpointManager(run_dir)

    processor, model = load_model(
        model_id_or_path=config.model_id,
        revision=config.revision,
        device=config.device,
        cache_dir=config.cache_dir,
    )
    train_loader, val_loader, test_loader = prepare_data_loaders(
        dataset_location=config.dataset,
        train_batch_size=config.batch_size,
        processor=processor,
        device=config.device,
        num_workers=config.num_workers,
        test_loaders_workers=config.val_num_workers,
    )
    peft_model = prepare_peft_model(
        model=model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        use_rslora=config.use_rslora,
        init_lora_weights=config.init_lora_weights,
        revision=config.revision,
    )
    training_metrics_tracker = MetricsTracker.init(metrics=["loss"])
    metrics = ["loss"]
    for metric in config.metrics:
        metrics += metric.describe()
    validation_metrics_tracker = MetricsTracker.init(metrics=metrics)

    run_training_loop(
        processor=processor,
        model=peft_model,
        data_loaders=(train_loader, val_loader),
        config=config,
        training_metrics_tracker=training_metrics_tracker,
        validation_metrics_tracker=validation_metrics_tracker,
        checkpoint_manager=checkpoint_manager,
    )

    save_metric_plots(
        training_tracker=training_metrics_tracker,
        validation_tracker=validation_metrics_tracker,
        output_dir=os.path.join(config.output_dir, "metrics"),
    )
    training_metrics_tracker.as_json(output_dir=os.path.join(config.output_dir, "metrics"), filename="training.json")
    validation_metrics_tracker.as_json(
        output_dir=os.path.join(config.output_dir, "metrics"), filename="validation.json"
    )

    # Log out paths for latest and best checkpoints
    print(f"Latest checkpoint saved at: {checkpoint_manager.latest_checkpoint_dir}")
    print(f"Best checkpoint saved at: {checkpoint_manager.best_checkpoint_dir}")


def run_training_loop(
    processor: AutoProcessor,
    model: PeftModel,
    data_loaders: tuple[DataLoader, Optional[DataLoader]],
    config: Configuration,
    training_metrics_tracker: MetricsTracker,
    validation_metrics_tracker: MetricsTracker,
    checkpoint_manager: CheckpointManager,
) -> None:
    train_loader, val_loader = data_loaders
    optimizer = get_optimizer(model=model, config=config)
    total_steps = config.epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )
    for epoch in range(config.epochs):
        run_training_epoch(
            processor=processor,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epoch=epoch + 1,
            config=config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            training_metrics_tracker=training_metrics_tracker,
            validation_metrics_tracker=validation_metrics_tracker,
            checkpoint_manager=checkpoint_manager,
        )


def run_training_epoch(
    processor: AutoProcessor,
    model: PeftModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epoch: int,
    config: Configuration,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    training_metrics_tracker: MetricsTracker,
    validation_metrics_tracker: MetricsTracker,
    checkpoint_manager: CheckpointManager,
) -> None:
    model.train()
    training_losses: list[float] = []

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{config.epochs}", unit="batch") as pbar:
        for step_id, batch in enumerate(train_loader):
            # TO DO : Need to check the batch
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss = loss.item()
            training_metrics_tracker.register(
                metric="loss",
                epoch=epoch,
                step=step_id + 1,
                value=loss,
            )
            training_losses.append(loss)

            # Update progress bar
            last_100_losses = training_losses[-100:]
            loss_moving_average = sum(last_100_losses) / len(last_100_losses) if last_100_losses else 0.0
            pbar.set_postfix({"Loss": f"{loss_moving_average:.4f}"})
            pbar.update(1)

    # Save checkpoints based on training loss if no validation loader
    if val_loader is None or len(val_loader) == 0:
        train_loss = sum(training_losses) / len(training_losses)
        checkpoint_manager.save_latest(processor, model)
        checkpoint_manager.save_best(processor, model, train_loss)
        return

    run_validation_epoch(
        processor=processor,
        model=model,
        loader=val_loader,
        epoch_number=epoch,
        config=config,
        metrics_tracker=validation_metrics_tracker,
    )

    val_loss = validation_metrics_tracker.get_metric_values("loss")[-1][2]
    checkpoint_manager.save_latest(processor, model)
    checkpoint_manager.save_best(processor, model, val_loss)


def run_validation_epoch(
    processor: AutoProcessor,
    model: Union[PeftModel, PaliGemmaForConditionalGeneration],
    loader: DataLoader,
    config: Configuration,
    metrics_tracker: MetricsTracker,
    epoch_number: int,
) -> None:
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(
                text=targets, return_tensors="pt", padding=True, return_token_type_ids=False
            ).input_ids.to(config.device)
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
        avg_val_loss = val_loss / len(loader)
        metrics_tracker.register(
            metric="loss",
            epoch=epoch_number,
            step=1,
            value=avg_val_loss,
        )
        # Run inference once for all metrics
        prompts, expected_responses, generated_texts, images = run_predictions(
            dataset=loader.dataset,
            processor=processor,
            model=model,
            device=config.device,
        )

        metrics_results = {"loss": avg_val_loss}

        for metric in config.metrics:
            if isinstance(metric, MeanAveragePrecisionMetric):
                classes = extract_unique_detection_dataset_classes(loader.dataset)
                targets, predictions = postprocess_paligemma_output_for_mean_average_precision(
                    expected_responses=expected_responses,
                    generated_texts=generated_texts,
                    images=images,
                    classes=classes,
                    processor=processor,
                )
                result = metric.compute(targets=targets, predictions=predictions)
                for key, value in result.items():
                    metrics_tracker.register(
                        metric=key,
                        epoch=epoch_number,
                        step=1,
                        value=value,
                    )
                    metrics_results[key] = value

        print("Validation Metrics:", ", ".join([f"{k}: {v:.4f}" for k, v in metrics_results.items()]))

        # Display inference results in IPython environments
        display_results(prompts, expected_responses, generated_texts, images)


def get_optimizer(model: PeftModel, config: Configuration) -> Optimizer:
    optimizer_type = config.optimizer.lower()
    if optimizer_type == "adamw":
        return AdamW(model.parameters(), lr=config.lr)
    if optimizer_type == "adam":
        return Adam(model.parameters(), lr=config.lr)
    if optimizer_type == "sgd":
        return SGD(model.parameters(), lr=config.lr)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def evaluate(config: Configuration) -> None:
    processor, model = load_model(
        model_id_or_path=config.model_id,
        revision=config.revision,
        device=config.device,
        cache_dir=config.cache_dir,
    )
    train_loader, val_loader, test_loader = prepare_data_loaders(
        dataset_location=config.dataset,
        train_batch_size=config.batch_size,
        processor=processor,
        device=config.device,
        num_workers=config.num_workers,
        test_loaders_workers=config.val_num_workers,
    )
    evaluation_loader = test_loader if test_loader is not None else val_loader

    metrics = []
    for metric in config.metrics:
        metrics += metric.describe()
    evaluation_metrics_tracker = MetricsTracker.init(metrics=metrics)

    # Run inference once for all metrics
    _, expected_responses, generated_texts, images = run_predictions(
        dataset=evaluation_loader.dataset,
        processor=processor,
        model=model,
        device=config.device,
    )

    for metric in config.metrics:
        if isinstance(metric, MeanAveragePrecisionMetric):
            classes = extract_unique_detection_dataset_classes(train_loader.dataset)
            targets, predictions = postprocess_paligemma_output_for_mean_average_precision(
                expected_responses=expected_responses,
                generated_texts=generated_texts,
                images=images,
                classes=classes,
                processor=processor,
            )
            result = metric.compute(targets=targets, predictions=predictions)
            for key, value in result.items():
                evaluation_metrics_tracker.register(
                    metric=key,
                    epoch=1,
                    step=1,
                    value=value,
                )

    evaluation_metrics_tracker.as_json(
        output_dir=os.path.join(config.output_dir, "metrics"), filename="evaluation.json"
    )
