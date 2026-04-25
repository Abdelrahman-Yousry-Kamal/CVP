#Phase 1: Diagnosis Module 
import numpy as np
import cv2
from scipy.ndimage import median_filter

def calculate_contrast(gray_img):
    """
    Calculate contrast using percentile-based range.    
    Args:
        gray_img: Grayscale image (numpy array, uint8)
    
    Returns:
        Contrast value (0-255). Below ~50 = low contrast (flat/washed out).
    """
    p95 = np.percentile(gray_img, 95)
    p5 = np.percentile(gray_img, 5)
    return p95 - p5


def calculate_exposure(gray_img):
    """
    Calculate exposure using mean brightness.
    
    Mean < 60       → Underexposed (too dark)
    Mean > 190      → Overexposed (too bright)
    60 to 190          → Acceptable
    
    Args:
        gray_img: Grayscale image (numpy array, uint8)
    
    Returns:
        Mean brightness value (0-255).
    """
    return np.mean(gray_img)


def calculate_exposure_median(gray_img):
    """
    Alternative exposure metric using median (better on images 
    with large dark/bright regions).
    
    Args:
        gray_img: Grayscale image (numpy array, uint8)
    
    Returns:
        Median brightness value (0-255).
    """
    return np.median(gray_img)


def calculate_noise(gray_img):
    """
    Calculate noise by measuring variance of residual after median filtering.
    
    Insight: A clean image looks nearly identical to its median-filtered version.
    The difference (residual) will be near zero. A noisy image has high-frequency 
    variation that the median filter removes — that difference will have high variance.
    
    Variance < 20   → Clean
    20–80           → Mild noise
    > 80            → Heavy noise (salt-and-pepper or Gaussian noise)
    
    Args:
        gray_img: Grayscale image (numpy array, uint8)
    
    Returns:
        Noise variance value.
    """
    filtered = median_filter(gray_img.astype(np.float64), size=3)
    residual = gray_img.astype(np.float64) - filtered
    return np.var(residual)


def calculate_sharpness(gray_img):
    """
    Calculate sharpness using Laplacian variance (edge detection).
    Variance > 500  → Sharp
    100–500         → Slightly soft
    < 100           → Blurry
    
    Args:
        gray_img: Grayscale image (numpy array, uint8)
    
    Returns:
        Laplacian variance value.
    """
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    return np.var(laplacian)


def diagnose(gray_img):
    return {
        "contrast": calculate_contrast(gray_img),
        "exposure": calculate_exposure(gray_img),
        "noise": calculate_noise(gray_img),
        "sharpness": calculate_sharpness(gray_img)
    }
