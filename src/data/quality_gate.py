"""
Optional Image-Quality Gate Module
====================================
IMPORTANT SCIENTIFIC REPRODUCIBILITY NOTE:
The routines in this module provide automated checks for optical blur (Laplacian variance),
illumination extremes, and minimum spatial resolution.

THIS QUALITY GATE WAS NOT PART OF THE REPORTED FINAL TRAINING PIPELINE.
All reported baseline and HELPix-R experimental results in the manuscript were produced on
the unpruned, leakage-controlled, exact-deduplicated dataset cohorts.
This module is provided strictly as a modular research utility for future work.
"""

from pathlib import Path
from typing import Dict, Any
from PIL import Image
import numpy as np
import cv2


def evaluate_image_quality(
    image_path: str | Path,
    blur_threshold: float = 100.0,
    min_resolution: int = 224,
    brightness_low: float = 20.0,
    brightness_high: float = 235.0,
) -> Dict[str, Any]:
    """
    Evaluate basic photographic quality heuristics for a skin-lesion image:
    - Blur via variance of the Laplacian
    - Mean grayscale brightness
    - Resolution bounds
    """
    path_str = str(image_path)
    try:
        with Image.open(path_str) as pil_img:
            w, h = pil_img.size
            gray = np.array(pil_img.convert('L'))
    except Exception as e:
        return {
            "path": path_str,
            "readable": False,
            "passed": False,
            "error": str(e)
        }

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(np.mean(gray))

    passed_resolution = (w >= min_resolution and h >= min_resolution)
    passed_blur = (laplacian_var >= blur_threshold)
    passed_brightness = (brightness_low <= mean_brightness <= brightness_high)

    passed_all = passed_resolution and passed_blur and passed_brightness

    return {
        "path": path_str,
        "readable": True,
        "width": w,
        "height": h,
        "laplacian_variance": laplacian_var,
        "mean_brightness": mean_brightness,
        "passed_resolution": passed_resolution,
        "passed_blur": passed_blur,
        "passed_brightness": passed_brightness,
        "passed_overall": passed_all,
    }
