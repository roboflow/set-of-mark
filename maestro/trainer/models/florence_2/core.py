import os
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal, Optional, Union

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, get_scheduler

from maestro.trainer.common.data_loaders.datasets import create_data_loaders
from maestro.trainer.common.utils.file_system import create_new_run_directory
from maestro.trainer.common.utils.metrics import (
    BaseMetric,
    MeanAveragePrecisionMetric,
    MetricsTracker,
    display_results,
    save_metric_plots,
)
from maestro.trainer.common.utils.reproducibility import make_it_reproducible
from maestro.trainer.models.florence_2.checkpoints import (
    DEFAULT_FLORENCE2_MODEL_ID,
    DEFAULT_FLORENCE2_MODEL_REVISION,
    DEVICE,
    CheckpointManager,
    load_model,
)
from maestro.trainer.models.florence_2.inference import run_predictions
from maestro.trainer.models.florence_2.loaders import collate_fn
from maestro.trainer.models.florence_2.metrics import (
    get_unique_detection_classes,
    process_output_for_detection_metric,
    process_output_for_text_metric,
)

LoraInitLiteral = Literal["gaussian", "olora", "pissa", "pissa_niter_[number of iters]", "loftq"]


@dataclass(frozen=True)
class Configuration:
    """Configuration for a Florence-2 model.

    This class encapsulates all the parameters needed for training a Florence-2 model,
    including dataset paths, model specifications, training hyperparameters, and output
    settings.

    Attributes:
        dataset (str): Path to the dataset used for training.
        model_id (str): Identifier for the Florence-2 model.
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
        metrics (List[BaseMetric]): List of metrics to track during training.
    """

    dataset: str
    model_id: str = DEFAULT_FLORENCE2_MODEL_ID
    revision: str = DEFAULT_FLORENCE2_MODEL_REVISION
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
    output_dir: str = "./training/florence2"
    metrics: list[BaseMetric] = field(default_factory=list)


def train(config: Configuration) -> None:
    """Train a Florence-2 model using the provided configuration.

    This function sets up the training environment, prepares the model and data loaders,
    and runs the training loop. It also handles metric tracking and checkpoint saving.

    Args:
        config (Configuration): The configuration object containing all necessary
            parameters for training.

    Returns:
        None

    Raises:
        ValueError: If an unsupported optimizer is specified in the configuration.
    """
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
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_location=config.dataset,
        train_batch_size=config.batch_size,
        train_collect_fn=partial(collate_fn, processor=processor, device=config.device),
        train_num_workers=config.num_workers,
        test_batch_size=config.val_batch_size,
        test_collect_fn=partial(collate_fn, processor=processor, device=config.device),
        test_num_workers=config.val_num_workers,
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


def prepare_peft_model(
    model: AutoModelForCausalLM,
    r: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    bias: Literal["none", "all", "lora_only"] = "none",
    inference_mode: bool = False,
    use_rslora: bool = True,
    init_lora_weights: Union[bool, LoraInitLiteral] = "gaussian",
    revision: str = DEFAULT_FLORENCE2_MODEL_REVISION,
) -> PeftModel:
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
        task_type="CAUSAL_LM",
        lora_dropout=lora_dropout,
        bias=bias,
        inference_mode=inference_mode,
        use_rslora=use_rslora,
        init_lora_weights=init_lora_weights,
        revision=revision,
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model.to(model.device)


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
    loss_values: list[float] = []
    progress_bar = tqdm(total=len(train_loader), desc=f"training {epoch}/{config.epochs}", unit="batch")
    with progress_bar:
        for batch_id, (inputs, _, answers, _) in enumerate(train_loader):
            labels = processor.tokenizer(
                text=answers, return_tensors="pt", padding=True, return_token_type_ids=False
            ).input_ids.to(config.device)
            outputs = model(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss = loss.item()

            training_metrics_tracker.register(
                metric="loss",
                epoch=epoch,
                step=batch_id + 1,
                value=loss,
            )
            loss_values.append(loss)
            average_loss = sum(loss_values) / len(loss_values) if loss_values else 0.0

            progress_bar.set_postfix({"loss": f"{average_loss: .4f}"})
            progress_bar.update(1)

    # Save checkpoints based on training loss if no validation loader
    if val_loader is None or len(val_loader) == 0:
        train_loss = sum(loss_values) / len(loss_values) if loss_values else 0.0
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
    model: Union[PeftModel, AutoModelForCausalLM],
    loader: DataLoader,
    config: Configuration,
    metrics_tracker: MetricsTracker,
    epoch_number: int,
) -> None:
    loss_values: list[float] = []
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="running validation", unit="batch")
        for inputs, questions, answers, images in progress_bar:
            labels = processor.tokenizer(
                text=answers, return_tensors="pt", padding=True, return_token_type_ids=False
            ).input_ids.to(config.device)
            outputs = model(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], labels=labels)
            loss_values.append(outputs.loss.item())
        average_loss = sum(loss_values) / len(loss_values) if loss_values else 0.0
        metrics_tracker.register(
            metric="loss",
            epoch=epoch_number,
            step=1,
            value=average_loss,
        )
        # Run inference once for all metrics
        questions, expected_answers, generated_answers, images = run_predictions(
            loader=loader, processor=processor, model=model
        )

        metrics_results = {"loss": average_loss}

        for metric in config.metrics:
            if isinstance(metric, MeanAveragePrecisionMetric):
                classes = get_unique_detection_classes(loader.dataset)
                targets, predictions = process_output_for_detection_metric(
                    expected_answers=expected_answers,
                    generated_answers=generated_answers,
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
            else:
                predictions = process_output_for_text_metric(
                    generated_answers=generated_answers,
                    images=images,
                    processor=processor,
                )
                result = metric.compute(predictions=predictions, targets=expected_answers)
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
        display_results(questions, expected_answers, generated_answers, images)


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
    """Evaluate a Florence-2 model using the provided configuration.

    This function loads the model and data, runs predictions on the evaluation dataset,
    computes specified metrics, and saves the results.

    Args:
        config (Configuration): The configuration object containing all necessary
            parameters for evaluation.

    Returns:
        None
    """
    processor, model = load_model(
        model_id_or_path=config.model_id,
        revision=config.revision,
        device=config.device,
        cache_dir=config.cache_dir,
    )
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_location=config.dataset,
        train_batch_size=config.batch_size,
        train_collect_fn=partial(collate_fn, processor=processor, device=config.device),
        train_num_workers=config.num_workers,
        test_batch_size=config.val_batch_size,
        test_collect_fn=partial(collate_fn, processor=processor, device=config.device),
        test_num_workers=config.val_num_workers,
    )
    evaluation_loader = test_loader if test_loader is not None else val_loader

    metrics = []
    for metric in config.metrics:
        metrics += metric.describe()
    evaluation_metrics_tracker = MetricsTracker.init(metrics=metrics)

    # Run inference once for all metrics
    _, expected_answers, generated_answers, images = run_predictions(
        loader=evaluation_loader, processor=processor, model=model
    )

    for metric in config.metrics:
        if isinstance(metric, MeanAveragePrecisionMetric):
            classes = get_unique_detection_classes(train_loader.dataset)
            targets, predictions = process_output_for_detection_metric(
                expected_answers=expected_answers,
                generated_answers=generated_answers,
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
        else:
            predictions = process_output_for_text_metric(
                generated_answers=generated_answers,
                images=images,
                processor=processor,
            )
            result = metric.compute(targets=expected_answers, predictions=predictions)
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
