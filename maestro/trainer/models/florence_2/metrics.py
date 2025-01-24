import re

import numpy as np
import supervision as sv
from PIL import Image
from transformers import AutoProcessor

from maestro.trainer.common.data_loaders.datasets import RoboflowJSONLDataset

DETECTION_CLASS_PATTERN = r"([a-zA-Z0-9 -]+)<loc_\d+>"


def process_output_for_detection_metric(
    expected_answers: list[str],
    generated_answers: list[str],
    images: list[Image.Image],
    classes: list[str],
    processor: AutoProcessor,
) -> tuple[list[sv.Detections], list[sv.Detections]]:
    targets = []
    predictions = []

    for image, suffix, generated_text in zip(images, expected_answers, generated_answers):
        # Postprocess prediction for mean average precision calculation
        prediction = processor.post_process_generation(generated_text, task="<OD>", image_size=image.size)
        prediction = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, prediction, resolution_wh=image.size)
        if len(prediction) == 0:
            prediction["class_name"] = []
        prediction = prediction[np.isin(prediction["class_name"], classes)]
        prediction.class_id = np.array([classes.index(class_name) for class_name in prediction["class_name"]])
        # Set confidence for mean average precision calculation
        prediction.confidence = np.ones(len(prediction))

        # Postprocess target for mean average precision calculation
        target = processor.post_process_generation(suffix, task="<OD>", image_size=image.size)
        if len(target) == 0:
            target["class_name"] = []
        target = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, target, resolution_wh=image.size)
        target.class_id = np.array([classes.index(class_name) for class_name in target["class_name"]])

        targets.append(target)
        predictions.append(prediction)

    return targets, predictions


def process_output_for_text_metric(
    generated_answers: list[str],
    images: list[Image.Image],
    processor: AutoProcessor,
) -> list[str]:
    predictions = []
    for image, generated_text in zip(images, generated_answers):
        prediction = processor.post_process_generation(generated_text, task="pure_text", image_size=image.size)[
            "pure_text"
        ]
        predictions.append(prediction)

    return predictions


def get_unique_detection_classes(dataset: RoboflowJSONLDataset) -> list[str]:
    class_set = set()
    for i in range(len(dataset)):
        image, data = dataset[i]
        suffix = data["suffix"]
        classes = re.findall(DETECTION_CLASS_PATTERN, suffix)
        class_set.update(classes)
    return sorted(class_set)
