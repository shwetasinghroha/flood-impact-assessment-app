from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from PIL import Image


IMAGE_SIZE = 512

TARGET_CLASS_NAMES = {
    0: "background_other",
    1: "flooded_building",
    2: "flooded_road",
    3: "water",
}

TARGET_CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (160, 150, 20),
    3: (61, 230, 250),
}

BACKGROUND_CLASS = 0
FLOODED_BUILDING_CLASS = 1
FLOODED_ROAD_CLASS = 2
WATER_CLASS = 3


def resize_image(image: np.ndarray, size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)) -> np.ndarray:
    pil_img = Image.fromarray(image)
    pil_img = pil_img.resize(size, resample=Image.BILINEAR)
    return np.array(pil_img)


def preprocess_image(pil_image: Image.Image, image_size: int = IMAGE_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        original_resized_uint8: HWC uint8 resized image
        model_input: CHW float32 normalized image
    """
    image = np.array(pil_image.convert("RGB"))
    image_resized = resize_image(image, size=(image_size, image_size))
    image_input = image_resized.astype(np.float32) / 255.0
    image_input = np.transpose(image_input, (2, 0, 1))
    return image_resized, image_input


def mask_to_rgb(mask: np.ndarray, color_dict: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_dict.items():
        rgb[mask == class_id] = np.array(color, dtype=np.uint8)
    return rgb


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color_dict: Dict[int, Tuple[int, int, int]],
    alpha: float = 0.40,
) -> np.ndarray:
    overlay = image.copy().astype(np.float32)

    for class_id, color in color_dict.items():
        if class_id == 0:
            continue
        region = mask == class_id
        color_arr = np.array(color, dtype=np.float32)
        overlay[region] = (1 - alpha) * overlay[region] + alpha * color_arr

    return np.clip(overlay, 0, 255).astype(np.uint8)


def compute_flood_indicators(mask: np.ndarray) -> Dict[str, float]:
    total_pixels = mask.size

    background_pixels = np.sum(mask == BACKGROUND_CLASS)
    flooded_building_pixels = np.sum(mask == FLOODED_BUILDING_CLASS)
    flooded_road_pixels = np.sum(mask == FLOODED_ROAD_CLASS)
    water_pixels = np.sum(mask == WATER_CLASS)

    indicators = {
        "background_other_pct": 100.0 * background_pixels / total_pixels,
        "flooded_building_pct": 100.0 * flooded_building_pixels / total_pixels,
        "flooded_road_pct": 100.0 * flooded_road_pixels / total_pixels,
        "water_pct": 100.0 * water_pixels / total_pixels,
    }

    indicators["combined_flooded_infra_pct"] = (
        indicators["flooded_building_pct"] + indicators["flooded_road_pct"]
    )
    indicators["combined_flood_relevant_pct"] = (
        indicators["water_pct"]
        + indicators["flooded_building_pct"]
        + indicators["flooded_road_pct"]
    )

    return indicators


def classify_severity_infrastructure_impact(indicators: Dict[str, float]) -> str:
    water = indicators["water_pct"]
    building = indicators["flooded_building_pct"]
    road = indicators["flooded_road_pct"]
    infra = indicators["combined_flooded_infra_pct"]

    if (
        infra >= 10
        or building >= 4
        or road >= 5
        or (water >= 30 and infra >= 3)
        or (water >= 20 and building >= 2)
        or (water >= 20 and road >= 2)
    ):
        return "Severe"
    elif (
        infra >= 2
        or building >= 1
        or road >= 1
        or water >= 12
    ):
        return "Moderate"
    else:
        return "Low"


def generate_evidence_summary(image_name: str, indicators: Dict[str, float], severity: str) -> str:
    return (
        f"Image {image_name}: predicted water coverage is {indicators['water_pct']:.2f}%, "
        f"flooded building coverage is {indicators['flooded_building_pct']:.2f}%, "
        f"and flooded road coverage is {indicators['flooded_road_pct']:.2f}%. "
        f"Combined flooded infrastructure coverage is {indicators['combined_flooded_infra_pct']:.2f}%. "
        f"Based on these visible flood indicators, the preliminary infrastructure-impact severity is classified as {severity}."
    )