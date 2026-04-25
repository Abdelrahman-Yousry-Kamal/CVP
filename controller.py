"""
Phase 2: Controller Module
Brain of the system. Reads metrics and returns an ordered treatment pipeline.

Order: EXPOSURE CORRECTION → NOISE REMOVAL → CONTRAST ENHANCEMENT → SHARPENING
"""

CONTRAST_LOW = 50
CONTRAST_MEDIUM = 100
CONTRAST_HIGH = 200
EXPOSURE_DARK = 60
EXPOSURE_BRIGHT = 190
NOISE_LOW = 20
NOISE_HIGH = 80
SHARPNESS_LOW = 100
SHARPNESS_MEDIUM = 500
SHARPNESS_HIGH = 1000


def prescribe(metrics):
    """
    Read metrics and return ordered treatment pipeline.

    Args:
        metrics: Dictionary from diagnose() with keys: contrast, exposure, noise, sharpness

    Returns:
        List of treatment function names to apply in order.
    """
    pipeline = []
    exposure_corrected = False

    # STEP 1: EXPOSURE CORRECTION
    if metrics["exposure"] < EXPOSURE_DARK:
        pipeline.append("gamma_correction_brighten")
        exposure_corrected = True
    elif metrics["exposure"] > EXPOSURE_BRIGHT:
        pipeline.append("gamma_correction_darken")
        exposure_corrected = True

    # STEP 2: NOISE REMOVAL
    if metrics["noise"] > NOISE_HIGH:
        pipeline.append("median_filter")
    elif metrics["noise"] > NOISE_LOW:
        pipeline.append("gaussian_filter")

    # STEP 3: CONTRAST ENHANCEMENT
    if metrics["contrast"] < CONTRAST_LOW:
        if exposure_corrected:
            pass  # Gamma already handles contrast — adding more will corrupt colors
        elif metrics["noise"] > NOISE_LOW:
            pipeline.append("contrast_stretching")
        else:
            pipeline.append("histogram_equalization")
    # STEP 4: SHARPENING (never on heavily noisy images)
    if metrics["sharpness"] < SHARPNESS_LOW and metrics["noise"] < NOISE_HIGH:
        pipeline.append("unsharp_mask")
    elif metrics["sharpness"] < SHARPNESS_MEDIUM and metrics["noise"] < NOISE_LOW:
        pipeline.append("laplacian_sharpen")

    return pipeline


def diagnose_and_prescribe(metrics):
    """
    Return treatment pipeline alongside a human-readable diagnostic report.

    Args:
        metrics: Dictionary from diagnose()

    Returns:
        Tuple of (treatment_pipeline, diagnostic_report_dict)
    """
    pipeline = prescribe(metrics)

    report = {
        "contrast_status": classify_contrast(metrics["contrast"]),
        "exposure_status": classify_exposure(metrics["exposure"]),
        "noise_status": classify_noise(metrics["noise"]),
        "sharpness_status": classify_sharpness(metrics["sharpness"]),
        "treatments": pipeline,
        "treatment_count": len(pipeline)
    }

    return pipeline, report


def classify_contrast(v):
    if v < CONTRAST_LOW:
        return "LOW"
    elif v > CONTRAST_HIGH:
        return "VERY_HIGH"
    return "ACCEPTABLE"


def classify_exposure(v):
    if v < EXPOSURE_DARK:
        return "UNDEREXPOSED"
    elif v > EXPOSURE_BRIGHT:
        return "OVEREXPOSED"
    return "ACCEPTABLE"


def classify_noise(v):
    if v < NOISE_LOW:
        return "CLEAN"
    elif v < NOISE_HIGH:
        return "MILD"
    return "HEAVY"


def classify_sharpness(v):
    if v < SHARPNESS_LOW:
        return "BLURRY"
    elif v < SHARPNESS_MEDIUM:
        return "SLIGHTLY_SOFT"
    return "SHARP"