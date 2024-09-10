import re
from typing import List, Dict
from typing import Tuple

import numpy as np
import supervision as sv
import torch
from PIL import Image
from supervision.metrics.mean_average_precision import MeanAveragePrecision
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

from maestro.trainer.common.data_loaders.datasets import DetectionDataset
from maestro.trainer.common.utils.metrics import BaseMetric

DETECTION_CLASS_PATTERN = r"([a-zA-Z0-9 -]+)<loc_\d+>"


class MeanAveragePrecisionMetric(BaseMetric):

    def describe(self) -> List[str]:
        return ["map50:95", "map50", "map75"]

    def compute(
        self,
        targets: List[sv.Detections],
        predictions: List[sv.Detections]
    ) -> Dict[str, float]:
        result = MeanAveragePrecision().update(targets=targets, predictions=predictions).compute()
        return {
            "map50:95": result.map50_95,
            "map50": result.map50,
            "map75": result.map75
        }


def postprocess_florence2_output_for_mean_average_precision(
    expected_responses: List[str],
    generated_texts: List[str],
    images: List[Image.Image],
    classes: List[str],
    processor: AutoProcessor
) -> Tuple[List[sv.Detections], List[sv.Detections]]:
    targets = []
    predictions = []
    
    for image, suffix, generated_text in zip(images, expected_responses, generated_texts):
        # Postprocess prediction for mean average precision calculation
        prediction = processor.post_process_generation(generated_text, task="<OD>", image_size=image.size)
        prediction = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, prediction, resolution_wh=image.size)
        prediction = prediction[np.isin(prediction["class_name"], classes)]
        prediction.class_id = np.array([classes.index(class_name) for class_name in prediction["class_name"]])
        # Set confidence for mean average precision calculation
        prediction.confidence = np.ones(len(prediction))
        
        # Postprocess target for mean average precision calculation
        target = processor.post_process_generation(suffix, task="<OD>", image_size=image.size)
        target = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, target, resolution_wh=image.size)
        target.class_id = np.array([classes.index(class_name) for class_name in target["class_name"]])
        
        targets.append(target)
        predictions.append(prediction)
    
    return targets, predictions


def run_predictions(
    dataset: DetectionDataset,
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
    device: torch.device,
) -> Tuple[List[str], List[str], List[str], List[Image.Image]]:
    prompts = []
    expected_responses = []
    generated_texts = []
    images = []
    
    for idx in tqdm(list(range(len(dataset))), desc="Generating predictions..."):
        image, data = dataset.dataset[idx]
        prefix = data["prefix"]
        suffix = data["suffix"]

        inputs = processor(text=prefix, images=image, return_tensors="pt").to(device)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        prompts.append(prefix)
        expected_responses.append(suffix)
        generated_texts.append(generated_text)
        images.append(image)
    
    return prompts, expected_responses, generated_texts, images


def extract_unique_detection_dataset_classes(dataset: DetectionDataset) -> List[str]:
    class_set = set()
    for i in range(len(dataset)):
        _, suffix, _ = dataset[i]
        classes = re.findall(DETECTION_CLASS_PATTERN, suffix)
        class_set.update(classes)
    return sorted(class_set)
