"""
Phase 2: Controller Module
This is the brain of the system. It reads the metrics and returns an ordered 
list of treatments to apply. Order matters
THE RULE OF ORDER:
EXPOSURE CORRECTION → NOISE REMOVAL → CONTRAST ENHANCEMENT → SHARPENING
"""

# Threshold Constants
CONTRAST_LOW = 50          # below this = low contrast
CONTRAST_HIGH = 200        # above this = leave it alone (very high contrast)
EXPOSURE_DARK = 60         # mean below this = underexposed
EXPOSURE_BRIGHT = 190      # mean above this = overexposed
NOISE_LOW = 20             # below = clean
NOISE_HIGH = 80            # above = heavily noisy (salt-and-pepper)
SHARPNESS_LOW = 100        # below = blurry
SHARPNESS_MEDIUM = 500     # below = slightly soft


def prescribe(metrics):
    """
    Decision logic: read metrics and return ordered treatment pipeline.
        Args:
        metrics: Dictionary from diagnose() with keys: contrast, exposure, noise, sharpness
    
    Returns:
        List of treatment function names to apply in order. Empty list for perfect images.
    """
    pipeline = []
    
    # STEP 1: EXPOSURE CORRECTION
    if metrics["exposure"] < EXPOSURE_DARK:
        pipeline.append("gamma_correction_brighten")
    elif metrics["exposure"] > EXPOSURE_BRIGHT:
        pipeline.append("gamma_correction_darken")
    
    # STEP 2: NOISE REMOVAL
    # Must happen before sharpening to avoid amplifying noise artifacts
    if metrics["noise"] > NOISE_HIGH:
        # Salt-and-pepper noise: use median filter (non-linear, removes sp without spreading)
        pipeline.append("median_filter")
    elif metrics["noise"] > NOISE_LOW:
        # Mild/Gaussian noise: use Gaussian filter (smoother, preserves edges better)
        pipeline.append("gaussian_filter")
    
    # STEP 3: CONTRAST ENHANCEMENT
    # Only apply if contrast is genuinely low
    if metrics["contrast"] < CONTRAST_LOW:
        # For noisy images, use contrast stretching (safer, less aggressive)
        # For clean images, use histogram equalization (more aggressive spread)
        if metrics["noise"] > NOISE_LOW:
            pipeline.append("contrast_stretching")
        else:
            pipeline.append("histogram_equalization")
    
    # STEP 4: SHARPENING — ONLY IF NOISE IS ACCEPTABLE
    # Never sharpen noisy images — you'll bake the noise artifacts in permanently
    if metrics["sharpness"] < SHARPNESS_LOW and metrics["noise"] < NOISE_HIGH:
        # Image is blurry AND not heavily noisy → safe to sharpen
        pipeline.append("unsharp_mask")
    elif metrics["sharpness"] < SHARPNESS_MEDIUM and metrics["noise"] < NOISE_LOW:
        # Image is slightly soft AND very clean → slightly aggressive sharpening
        pipeline.append("laplacian_sharpen")
    
    return pipeline


def diagnose_and_prescribe(metrics):
    """
    Generate a readable diagnostic report alongside the treatment pipeline.
    
    Useful for logging and understanding what decisions were made.
    
    Args:
        metrics: Dictionary from diagnose()
    
    Returns:
        Tuple of (treatment_pipeline, diagnostic_report_dict)
    """
    pipeline = prescribe(metrics)
    
    # Build descriptive report
    report = {
        "contrast_status": classify_contrast(metrics["contrast"]),
        "exposure_status": classify_exposure(metrics["exposure"]),
        "noise_status": classify_noise(metrics["noise"]),
        "sharpness_status": classify_sharpness(metrics["sharpness"]),
        "treatments": pipeline,
        "treatment_count": len(pipeline)
    }
    
    return pipeline, report


def classify_contrast(contrast_value):
    """Classify contrast metric into human-readable category."""
    if contrast_value < CONTRAST_LOW:
        return "LOW"
    elif contrast_value > CONTRAST_HIGH:
        return "VERY_HIGH"
    else:
        return "ACCEPTABLE"


def classify_exposure(exposure_value):
    """Classify exposure metric into human-readable category."""
    if exposure_value < EXPOSURE_DARK:
        return "UNDEREXPOSED"
    elif exposure_value > EXPOSURE_BRIGHT:
        return "OVEREXPOSED"
    else:
        return "ACCEPTABLE"


def classify_noise(noise_value):
    """Classify noise metric into human-readable category."""
    if noise_value < NOISE_LOW:
        return "CLEAN"
    elif noise_value < NOISE_HIGH:
        return "MILD"
    else:
        return "HEAVY"


def classify_sharpness(sharpness_value):
    """Classify sharpness metric into human-readable category."""
    if sharpness_value < SHARPNESS_LOW:
        return "BLURRY"
    elif sharpness_value < SHARPNESS_MEDIUM:
        return "SLIGHTLY_SOFT"
    else:
        return "SHARP"
