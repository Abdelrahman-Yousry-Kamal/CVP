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
# GAMMA CORRECTION (Exposure Adjustment)


def gamma_correction(img, gamma):
    """
    Apply gamma correction to an image.
    
    Gamma correction preserves the relative tonal relationships in an image.
    It's more natural-looking than simple multiplication (which clips 
    highlights aggressively).
    
    Args:
        img: Input image (BGR or grayscale)
        gamma: Gamma value
               gamma < 1 = brighten (lighten shadows more than highlights)
               gamma > 1 = darken (darken highlights more than shadows)
               gamma = 2.0 = significant darkening
               gamma = 0.5 = significant brightening
    
    Returns:
        Gamma-corrected image (same type as input)
    """
    # Build lookup table: for each input intensity i, compute (i/255)^(1/gamma) * 255
    table = np.array([(i / 255.0) ** (1.0 / gamma) * 255
                      for i in range(256)]).astype(np.uint8)
    
    # Apply lookup table to all channels
    return cv2.LUT(img, table)


def gamma_correction_brighten(img):
    """Brighten image (gamma = 1.8). Use for underexposed images."""
    return gamma_correction(img, gamma=1.8)


def gamma_correction_darken(img):
    """Darken image (gamma = 0.6). Use for overexposed images."""
    return gamma_correction(img, gamma=0.6)



# NOISE FILTERS


def median_filter(img, ksize=3):
    """
    Apply median filter (non-linear filter).
    
    Ideal for salt-and-pepper noise. Completely removes salt-and-pepper 
    without spreading it like Gaussian blur would.
    
    Args:
        img: Input image (BGR or grayscale)
        ksize: Kernel size (must be odd). Default=3 (3x3 kernel).
               Larger values remove more noise but blur more.
    
    Returns:
        Median-filtered image
    """
    return cv2.medianBlur(img, ksize)


def gaussian_filter(img, ksize=5, sigma=1.0):
    """
    Apply Gaussian blur filter.
    
    Better for Gaussian/random noise than median. Smoother results and 
    preserves edges better than box blur, but less aggressive at 
    removing salt-and-pepper than median.
    
    Args:
        img: Input image (BGR or grayscale)
        ksize: Kernel size (must be odd). Default=5 (5x5 kernel).
        sigma: Standard deviation of Gaussian kernel. Default=1.0.
               Larger values produce more blur.
    
    Returns:
        Gaussian-filtered image
    """
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)



# CONTRAST ENHANCEMENT


def contrast_stretching(img):
    """
    Contrast stretching (percentile-based normalization).
    
    Remaps the image intensity range from [p2, p98] to [0, 255]. This is 
    safer on noisy images than histogram equalization because it doesn't 
    spread noise across the full histogram.
    
    Working principle:
    - Find 2nd and 98th percentiles to exclude extreme outliers
    - Linearly stretch that range to [0, 255]
    - Clip values outside [0, 255] to bounds
    
    Args:
        img: Input image (BGR or grayscale)
    
    Returns:
        Contrast-stretched image
    """
    # Handle both grayscale and color images
    if len(img.shape) == 3:
        # For color images, apply to each channel
        channels = cv2.split(img)
        stretched_channels = []
        for channel in channels:
            p2 = np.percentile(channel, 2)
            p98 = np.percentile(channel, 98)
            stretched = np.clip((channel.astype(np.float64) - p2) / (p98 - p2 + 1e-6) * 255,
                               0, 255)
            stretched_channels.append(stretched.astype(np.uint8))
        return cv2.merge(stretched_channels)
    else:
        # Grayscale
        p2 = np.percentile(img, 2)
        p98 = np.percentile(img, 98)
        stretched = np.clip((img.astype(np.float64) - p2) / (p98 - p2 + 1e-6) * 255,
                           0, 255)
        return stretched.astype(np.uint8)


def histogram_equalization(img):
    """
    Histogram equalization for more aggressive contrast enhancement.
    
    CRITICAL: For color images, NEVER equalize R, G, B channels independently 
    — it destroys color balance. Instead:
    1. Convert to YCrCb (luminance + chrominance)
    2. Equalize only the Y (luminance) channel
    3. Convert back to BGR
    
    This preserves hue and saturation while enhancing brightness contrast.
    
    Args:
        img: Input image (BGR or grayscale)
    
    Returns:
        Histogram-equalized image
    """
    if len(img.shape) == 3:
        # Color image: work in YCrCb to preserve color
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # Equalize only the Y (luminance) channel
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        # Convert back to BGR
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        # Grayscale: direct equalization
        return cv2.equalizeHist(img)



# SHARPENING


def unsharp_mask(img, ksize=5, sigma=1.0, amount=1.5):
    """
    Unsharp masking: the standard sharpening technique.
    
    Algorithm:
    1. Create a blurred version of the image
    2. Compute high-frequency detail: original - blurred
    3. Add scaled high-frequency detail back to original
    
    Result: original + amount * (original - blurred)
    
    Advantages:
    - Tunable amount parameter for fine control
    - Produces natural-looking sharpening
    - Works well on photographs
    
    Args:
        img: Input image (BGR or grayscale)
        ksize: Gaussian blur kernel size (must be odd). Default=5.
        sigma: Gaussian blur standard deviation. Default=1.0.
        amount: Strength of sharpening (0.0-3.0 typical).
                Default=1.5. Higher = more aggressive.
    
    Returns:
        Sharpened image (clipped to [0, 255])
    """
    # Create blurred version
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    
    # Compute sharpened: orig + amount * (orig - blurred)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    
    # Clip to valid range [0, 255]
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def laplacian_sharpen(img, weight=0.7):
    """
    Laplacian sharpening    
    Algorithm:
    1. Compute Laplacian (second-derivative edge detector)
    2. Normalize Laplacian to [0, 255]
    3. Add scaled Laplacian back to original
    
    Result: original + weight * Laplacian
    
    Advantages:
    - More aggressive than unsharp mask
    - Better for technical/document images
    - Directly emphasizes edges
    
    Disadvantages:
    - Can create halos around edges if weight is too high
    - Should only be applied to clean images (no noise)
    
    Args:
        img: Input image (BGR or grayscale)
        weight: Strength of Laplacian sharpening (0.0-1.0 typical).
                Default=0.7. Higher = more aggressive.
    
    Returns:
        Sharpened image (clipped to [0, 255])
    """
    if len(img.shape) == 3:
        # For color images, work on grayscale to compute Laplacian
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Compute Laplacian (edge detection via second derivative)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Normalize Laplacian to [0, 255] range using absolute value
    lap_normalized = cv2.convertScaleAbs(lap)
    
    # Apply sharpening based on image type
    if len(img.shape) == 3:
        # For color image, convert normalized Laplacian back to BGR
        lap_bgr = cv2.cvtColor(lap_normalized, cv2.COLOR_GRAY2BGR)
        # Add scaled Laplacian detail
        sharpened = cv2.addWeighted(img, 1.0, lap_bgr, weight, 0)
    else:
        # For grayscale, add directly
        sharpened = cv2.addWeighted(img, 1.0, lap_normalized, weight, 0)
    
    # Clip to valid range [0, 255]
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# Maps treatment names (used in controller.py) to actual functions
TREATMENT_MAP = {
    # Exposure
    "gamma_correction_brighten": gamma_correction_brighten,
    "gamma_correction_darken": gamma_correction_darken,
    
    # Noise removal
    "median_filter": median_filter,
    "gaussian_filter": gaussian_filter,
    
    # Contrast
    "contrast_stretching": contrast_stretching,
    "histogram_equalization": histogram_equalization,
    
    # Sharpening
    "unsharp_mask": unsharp_mask,
    "laplacian_sharpen": laplacian_sharpen,
}
