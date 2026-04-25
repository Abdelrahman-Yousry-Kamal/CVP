"""
Phase 3: Treatment Module
All image enhancement functions. These are the actual transformations applied
to the image based on the prescription from controller.py.

Four categories of treatments:
1. Gamma Correction (exposure adjustment)
2. Noise Filters (median and Gaussian)
3. Contrast Enhancement (stretching and equalization)
4. Sharpening (unsharp mask and Laplacian)
"""

import numpy as np
import cv2


# --- GAMMA CORRECTION ---

def gamma_correction(img, gamma):
    """
    Args:
        gamma < 1 = brighten, gamma > 1 = darken
    """
    table = np.array([(i / 255.0) ** (1.0 / gamma) * 255
                      for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def gamma_correction_brighten(img):
    """Brighten image (gamma = 1.8). Use for underexposed images."""
    return gamma_correction(img, gamma=1.8)


def gamma_correction_darken(img):
    """Darken image (gamma = 0.6). Use for overexposed images."""
    return gamma_correction(img, gamma=0.6)


# --- NOISE FILTERS ---

def median_filter(img, ksize=3):
    """Median filter for salt-and-pepper noise. ksize=3 (mild)."""
    return cv2.medianBlur(img, ksize)


def median_filter_strong(img):
    """Stronger median filter for heavy noise. ksize=5."""
    return cv2.medianBlur(img, 5)


def gaussian_filter(img, ksize=5, sigma=1.0):
    """Gaussian blur for mild/random noise."""
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


# --- CONTRAST ENHANCEMENT ---

def contrast_stretching(img):
    """
    Percentile-based contrast stretching.

    For color images, operates on the Y (luminance) channel in YCrCb space
    to avoid shifting color balance. Grayscale is stretched directly.
    """
    if len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float64)
        p2, p98 = np.percentile(y, 2), np.percentile(y, 98)
        ycrcb[:, :, 0] = np.clip((y - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
        return np.clip((img.astype(np.float64) - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)


def histogram_equalization(img):
    """
    Histogram equalization.

    For color images, equalizes only the Y (luminance) channel in YCrCb
    to preserve hue and saturation. Never equalizes RGB channels independently.
    """
    if len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return cv2.equalizeHist(img)


# --- SHARPENING ---

def unsharp_mask(img, ksize=5, sigma=1.0, amount=1.5):
    """
    Unsharp masking: original + amount * (original - blurred).
    Natural-looking sharpening, works well on photographs.
    """
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def laplacian_sharpen(img, weight=0.7):
    """
    Laplacian sharpening: original + weight * Laplacian.
    More aggressive than unsharp mask. Only use on clean images — amplifies noise.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_normalized = cv2.convertScaleAbs(lap)

    if len(img.shape) == 3:
        lap_bgr = cv2.cvtColor(lap_normalized, cv2.COLOR_GRAY2BGR)
        sharpened = cv2.addWeighted(img, 1.0, lap_bgr, weight, 0)
    else:
        sharpened = cv2.addWeighted(img, 1.0, lap_normalized, weight, 0)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


# --- TREATMENT MAP ---

TREATMENT_MAP = {
    "gamma_correction_brighten": gamma_correction_brighten,
    "gamma_correction_darken":   gamma_correction_darken,
    "median_filter":             median_filter,
    "median_filter_strong":      median_filter_strong,
    "gaussian_filter":           gaussian_filter,
    "contrast_stretching":       contrast_stretching,
    "histogram_equalization":    histogram_equalization,
    "unsharp_mask":              unsharp_mask,
    "laplacian_sharpen":         laplacian_sharpen,
}