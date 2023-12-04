import re
from typing import Optional, Tuple, List

import numpy as np
import supervision as sv

from maestro.postprocessing.mask import (
    adjust_mask_features_by_relative_area,
    FeatureType,
    mask_non_max_suppression
)
from maestro.pipelines.base import BasePromptCreator, BaseResponseProcessor
from maestro.wrappers.sam import SegmentAnything


class SamPromptCreator(BasePromptCreator):
    def __init__(
        self,
        device: str = 'cpu',
        model_name: str = "facebook/sam-vit-huge",
        maximum_hole_area: float = 0.01,
        maximum_island_area: float = 0.01,
        minimum_mask_area: float = 0.02,
        maximum_mask_area: float = 1.0
    ) -> None:
        self.model = SegmentAnything(device=device, model_name=model_name)
        self.maximum_hole_area = maximum_hole_area
        self.maximum_island_area = maximum_island_area
        self.minimum_mask_area = minimum_mask_area
        self.maximum_mask_area = maximum_mask_area
        self.iou_threshold = 0.5

    @staticmethod
    def annotate(image: np.ndarray, marks: sv.Detections) -> np.ndarray:
        h, w, _ = image.shape
        text_scale = sv.calculate_dynamic_text_scale(resolution_wh=(w, h))
        line_thickness = sv.calculate_dynamic_line_thickness(resolution_wh=(w, h))
        label_annotator = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            text_scale=text_scale,
            color=sv.Color.black(),
            text_color=sv.Color.white(),
            text_position=sv.Position.CENTER_OF_MASS)
        polygon_annotator = sv.PolygonAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            thickness=line_thickness)

        labels = list(map(str, range(len(marks))))

        annotated_image = image.copy()
        annotated_image = polygon_annotator.annotate(
            scene=annotated_image, detections=marks)
        return label_annotator.annotate(
            scene=annotated_image, detections=marks, labels=labels)

    def refine(self, marks: sv.Detections) -> sv.Detections:
        total_area = marks.mask.shape[1] * marks.mask.shape[2]
        masks = []
        for mask in marks.mask:
            mask = adjust_mask_features_by_relative_area(
                mask=mask,
                area_threshold=self.maximum_island_area,
                feature_type=FeatureType.ISLAND)
            mask = adjust_mask_features_by_relative_area(
                mask=mask,
                area_threshold=self.maximum_hole_area,
                feature_type=FeatureType.HOLE)
            is_small_enough = np.sum(mask) / total_area >= self.minimum_mask_area
            is_large_enough = np.sum(mask) / total_area <= self.maximum_mask_area
            if np.any(mask) and is_small_enough and is_large_enough:
                masks.append(mask)
        masks = np.array(masks)
        masks = mask_non_max_suppression(masks=masks, iou_threshold=self.iou_threshold)
        return sv.Detections(
            mask=masks,
            xyxy=sv.mask_to_xyxy(masks=masks)
        )

    def create(
        self,
        text: str,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[str, np.ndarray, sv.Detections]:
        marks = self.model.predict(image=image, mask=mask)
        marks = self.refine(marks)
        annotated_image = self.annotate(image=image, marks=marks)
        return text, annotated_image, marks


class SamResponseProcessor(BaseResponseProcessor):

    @staticmethod
    def extract_mark_ids(text: str) -> List[str]:
        """
        Extracts all unique marks enclosed in square brackets from a given string.
            Duplicates are removed and the results are sorted in descending order.

        Args:
            text (str): The string to be searched.

        Returns:
            List[str]: A list of unique marks found within square brackets, sorted in
                descending order.
        """
        pattern = r'\[(\d+)\]'
        found_marks = re.findall(pattern, text)
        unique_marks = set(found_marks)
        return sorted(unique_marks, key=int, reverse=False)

    def process(self, text: str, marks: sv.Detections) -> sv.Detections:
        mark_ids = self.extract_mark_ids(text=text)
        mark_ids = np.array(mark_ids, dtype=int)
        return marks[mark_ids]

    @staticmethod
    def annotate(image: np.ndarray, marks: sv.Detections) -> np.ndarray:
        h, w, _ = image.shape
        line_thickness = sv.calculate_dynamic_line_thickness(resolution_wh=(w, h))
        mask_annotator = sv.MaskAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            opacity=0.4)
        polygon_annotator = sv.PolygonAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            thickness=line_thickness)

        annotated_image = image.copy()
        annotated_image = mask_annotator.annotate(
            scene=annotated_image, detections=marks)
        return polygon_annotator.annotate(
            scene=annotated_image, detections=marks)

    def visualize(
        self,
        text: str,
        image: np.ndarray,
        marks: sv.Detections
    ) -> np.ndarray:
        marks = self.process(text=text, marks=marks)
        return self.annotate(image=image, marks=marks)