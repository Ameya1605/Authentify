"""
Phase 2 — Image Preprocessing Module
Converts uploaded documents to standardized format for analysis.
- Grayscale conversion
- Noise removal
- Thresholding
- Resize & normalization
"""

import cv2
import numpy as np
from PIL import Image
import io


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """Load image from raw bytes into OpenCV format."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from uploaded file.")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale for uniform pixel analysis."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def remove_noise(img: np.ndarray, method: str = "gaussian") -> np.ndarray:
    """Apply noise removal to improve clarity."""
    if method == "gaussian":
        return cv2.GaussianBlur(img, (5, 5), 0)
    elif method == "median":
        return cv2.medianBlur(img, 5)
    elif method == "bilateral":
        return cv2.bilateralFilter(img, 9, 75, 75)
    elif method == "nlm":
        if len(img.shape) == 2:
            return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return img


def apply_threshold(img_gray: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """Apply thresholding for binarization."""
    if method == "otsu":
        _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    elif method == "adaptive":
        return cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    elif method == "simple":
        _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        return thresh
    return img_gray


def resize_normalize(img: np.ndarray, target_width: int = 1200) -> np.ndarray:
    """Resize & normalize image dimensions for consistent module performance."""
    h, w = img.shape[:2]
    if w <= 0:
        return img
    aspect = h / w
    new_h = int(target_width * aspect)
    resized = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)
    return resized


def preprocess_pipeline(file_bytes: bytes) -> dict:
    """
    Full preprocessing pipeline.
    Returns dict with original, grayscale, denoised, thresholded, and normalized images.
    """
    original = load_image_from_bytes(file_bytes)
    normalized = resize_normalize(original)
    grayscale = to_grayscale(normalized)
    denoised = remove_noise(grayscale, method="bilateral")
    thresholded = apply_threshold(denoised, method="adaptive")

    return {
        "original": original,
        "normalized": normalized,
        "grayscale": grayscale,
        "denoised": denoised,
        "thresholded": thresholded,
    }
