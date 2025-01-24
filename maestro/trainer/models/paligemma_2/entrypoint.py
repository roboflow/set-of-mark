import typer

from maestro.trainer.common.utils.metrics import (
    BaseMetric,
    BLEUMetric,
    CharacterErrorRateMetric,
    MeanAveragePrecisionMetric,
    TranslationErrorRateMetric,
    WordErrorRateMetric,
)

florence_2_app = typer.Typer(help="Fine-tune and evaluate PaliGemma 2 model")


METRIC_CLASSES: dict[str, type[BaseMetric]] = {
    MeanAveragePrecisionMetric.name: MeanAveragePrecisionMetric,
    WordErrorRateMetric.name: WordErrorRateMetric,
    CharacterErrorRateMetric.name: CharacterErrorRateMetric,
    BLEUMetric.name: BLEUMetric,
    TranslationErrorRateMetric.name: TranslationErrorRateMetric,
}


def parse_metrics(metrics: list[str]) -> list[BaseMetric]:
    metric_objects = []
    for metric_name in metrics:
        metric_class = METRIC_CLASSES.get(metric_name.lower())
        if metric_class:
            metric_objects.append(metric_class())
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")
    return metric_objects
