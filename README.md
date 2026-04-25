# Mini Project 2: Image Enhancement Pipeline

A complete image processing pipeline that diagnoses image quality issues and applies enhancements.

## Overview

This project implements a three-phase quality enhancement system:

1. **Phase 1 (Diagnosis)**: Measure image quality using four mathematically precise metrics
2. **Phase 2 (Controller)**: Prescribe optimal treatment sequence based on metrics  
3. **Phase 3 (Treatment)**: Apply enhancements in correct order
4. **Phase 4 (Gallery)**: Generate benchmarks showing before/after/histogram

### Key Principles

- **Order Matters**: EXPOSURE → NOISE → CONTRAST → SHARPENING 
- **Math-First**: All metrics are defensible and well-founded
- **Zero ML**: Pure numpy/cv2, no deep learning or sklearn
- **No Arbitrary Magic**: Every parameter is justified and documented
- **Multi-Format Support**: Handles JPEG, PNG, BMP, TIFF, GIF, WebP, AVIF, HEIC/HEIF, and more

The pipeline uses a robust **dual-loading strategy**:
1. **OpenCV first** (fast, standard)
2. **PIL fallback** (better format support, handles modern formats)

This ensures maximum compatibility without requiring format-specific dependencies.

## Project Structure

```
Mini_Project_2/
│
├── main.py                  # Entry point — runs full pipeline
├── diagnosis.py             # Phase 1: Metric calculations
├── controller.py            # Phase 2: Decision logic + thresholds
├── treatment.py             # Phase 3: Enhancement functions
├── utils.py                 # Helpers: loading, saving, logging
├── gallery_generator.py     # Build benchmark gallery visualization
├── dataset_generator.py     # (Helper) Create test dataset
│
├── dataset/
│   ├── input/               # Input images 
│   └── output/              # Enhanced images (auto-generated)
│
├── gallery/
│   └── benchmark_grid.png   # Side-by-side comparison grid
│
└── logs/
    └── diagnostics.json     # Per-image metrics + treatments
```

## Phase 1: Diagnosis Module (`diagnosis.py`)

Four metrics measuring distinct image quality aspects:

### Metric 1: Contrast
```python
contrast = p95 - p5  # 95th percentile - 5th percentile
```
- **Why**: Avoids outlier sensitivity of min-max range
- **Interpretation**: 
  - < 50 = flat/washed out
  - 50-200 = acceptable range
  - > 200 = very high contrast

### Metric 2: Exposure
```python
exposure = mean(grayscale_img)  # or median() for robustness
```
- **Interpretation**:
  - < 60 = underexposed (too dark)
  - 60-190 = acceptable
  - > 190 = overexposed (too bright)

### Metric 3: Noise
```python
residual = img - median_filter(img, size=3)
noise = var(residual)
```
- **Why**: Median filter removes noise without blurring edges
- **Interpretation**:
  - < 20 = clean
  - 20-80 = mild noise
  - > 80 = heavy (salt-and-pepper) noise

### Metric 4: Sharpness
```python
laplacian_var = var(cv2.Laplacian(img))
```
- **Why**: Edge detection via second derivative
- **Interpretation**:
  - < 100 = blurry
  - 100-500 = slightly soft
  - > 500 = sharp

## Phase 2: Controller (`controller.py`)

**The Cardinal Rule of Treatment Order**:
```
EXPOSURE CORRECTION → NOISE REMOVAL → CONTRAST ENHANCEMENT → SHARPENING
```

### Why This Order?

1. **Fix exposure first** — Subsequent metrics depend on correct baseline brightness
2. **Remove noise before sharpening** — Sharpening amplifies noise; apply first to prevent baking artifacts
3. **Contrast after denoising** — Histogram operations on noise spread it everywhere
4. **Sharpen last** — Only sharpen real edges, not noise

### Decision Logic

Threshold constants (top of `controller.py`):
```python
CONTRAST_LOW = 50              # below = needs enhancement
EXPOSURE_DARK = 60             # below = underexposed
EXPOSURE_BRIGHT = 190          # above = overexposed
NOISE_LOW = 20                 # below = clean
NOISE_HIGH = 80                # above = heavily noisy
SHARPNESS_LOW = 100            # below = blurry
```

Example prescription:
```
Dark image (60) + noisy (50) + low contrast (40) + blurry (80)
→ [gamma_correction_brighten, gaussian_filter, contrast_stretching]
   (No sharpening because image is noisy!)
```

## Phase 3: Treatment Module (`treatment.py`)

### Exposure Correction

```python
gamma_correction(img, gamma=1.8)   # gamma < 1 = darken
gamma_correction(img, gamma=0.6)   # gamma > 1 = brighten
```
- Preserves tonal relationships (more natural than simple multiplication)
- 1.8 = brightening, 0.6 = darkening

### Noise Removal

```python
median_filter(img, ksize=3)        # Salt-and-pepper (non-linear)
gaussian_filter(img, ksize=5)      # Gaussian/random noise (smooth)
```
- Median: Ideal for impulse noise, removes salt-and-pepper completely
- Gaussian: Smoother, preserves edges, better for natural Gaussian noise

### Contrast Enhancement

```python
contrast_stretching(img)           # Safe: percentile-based (2%, 98%)
histogram_equalization(img)        # Aggressive: full histogram spread
```
- Stretching: Safer on noisy images (doesn't amplify noise into extremes)
- Equalization: Better on clean images (maximum contrast recovery)
- **Critical**: For color images, equalize only Y channel (YCrCb space), preserve CrCb

### Sharpening

```python
unsharp_mask(img, ksize=5, amount=1.5)   # Tunable, natural-looking
laplacian_sharpen(img, weight=0.7)       # Aggressive, edge-focused
```
- Unsharp: Almost always preferred (photos, natural images)
- Laplacian: Better for technical/document images
- **Warning**: Only on clean images (noise < NOISE_HIGH)

## Phase 4: Gallery Generator (`gallery_generator.py`)

Generates a 3-column benchmark grid:

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Original | Histogram | Enhanced |

**Why 3 columns?**
- Visually prove histogram equalization spread the histogram
- Show edge recovery from sharpening
- Direct before/after comparison for grader

## How to Use

### 1. Install Dependencies

```bash
pip install numpy opencv-python scipy pillow matplotlib
```

### 2. Prepare Test Images

Option A: Use the dataset generator to create synthetic degradations
```bash
# First, place 3-4 clean reference images in dataset/input_base/
python dataset_generator.py --base-dir dataset/input_base --output-dir dataset/input
```

Option B: Manually place 15-20 images in `dataset/input/`

### 3. Run the Pipeline

```bash
python main.py
```

Output:
- Enhanced images → `dataset/output/`
- Diagnostics log → `logs/diagnostics.json`
- Benchmark gallery → `gallery/benchmark_grid.png`

### 4. Check Results

Open `gallery/benchmark_grid.png` to see before/after comparisons.

Read `logs/diagnostics.json` for per-image metrics:
```json
{
  "image_filename": "photo_01.jpg",
  "metrics": {
    "contrast": 142.5,
    "exposure": 88.3,
    "noise": 15.2,
    "sharpness": 245.8
  },
  "metric_classes": {
    "contrast_status": "ACCEPTABLE",
    "exposure_status": "ACCEPTABLE",
    "noise_status": "CLEAN",
    "sharpness_status": "SLIGHTLY_SOFT"
  },
  "treatments_applied": ["laplacian_sharpen"],
  "treatment_count": 1
}
```

## Test Coverage Strategy

Generate or acquire images covering all test scenarios:

| Count | Category | Notes |
|-------|----------|-------|
| 3 | Perfect/reference | Clean photos (pass-through) |
| 3 | Underexposed | `img * 0.2` or dimmer |
| 3 | Overexposed | `img * 2.0` or clipped |
| 3 | Salt-and-pepper | 4% random black/white pixels |
| 3 | Gaussian noise | Random normal(μ=0, σ=30) |
| 2 | Blurry | GaussianBlur(25x25) |
| 2 | Multiple defects | Dark+noisy, blurry+low contrast |

**Total**: 19 images to stress-test all code paths.

## Critical Design Decisions

### 1. Why Percentile Contrast, Not Min-Max?

Single outlier would break min-max:
```
Min-max contrast of a mostly gray image with one white pixel = 255
Percentile contrast = actual gray range ≈ 50-60
```

### 2. Why Median Filtering for Noise Measurement?

```
Noisy image - median(noisy) = residual variance (HIGH)
Clean image - median(clean) = nearly zero (LOW)
```
Median filter removes noise without spreading it (unlike box/Gaussian).

### 3. Why Laplacian for Sharpness?

- Edges have high second derivatives → high variance
- Blur flattens edges → low second derivatives
- Universal blur-detection metric (LAPLACIAN-VARIANCE method)

### 4. Why YCrCb for Histogram Equalization?

```
Bad: cv2.equalizeHist() on R,G,B separately
  → Channels shift independently → color destroyed

Good: Convert BGR → YCrCb, equalize Y, convert back
  → Y (brightness) enhanced, CrCb (color) preserved
```

### 5. Why Unsharp Mask Over Laplacian Sharpening?

- Unsharp: Tunable `amount` parameter → fine control → natural results
- Laplacian: Fixed weight → aggressive → halos if too strong
- For photos: Unsharp wins. For documents: Laplacian wins.

## Extending the Project

### Add a New Metric
1. Add function to `diagnosis.py`
2. Add to `diagnose()` return dict
3. Add threshold to `controller.py`
4. Add decision logic to `prescribe()`

### Add a New Treatment
1. Add function to `treatment.py` following naming convention
2. Add entry to `TREATMENT_MAP` at bottom
3. Add decision to `prescribe()` in `controller.py`
4. Document rationale in comments

### Custom Gallery Layout
Modify `build_gallery()` in `gallery_generator.py`:
```python
# Add 4th column for edge map, etc.
fig, axes = plt.subplots(num_images, 4, ...)
```

## Grading Checklist

- [ ] All 4 metrics implemented mathematically correctly
- [ ] Controller follows cardinal rule (exposure → noise → contrast → sharpen)
- [ ] Threshold values are reasonable and justified
- [ ] Color space handling correct (YCrCb for histogram equalization)
- [ ] Edge case handling (noisy+blurry → denoise only, perfect → []
- [ ] Gallery generated with clear before/after/histogram
- [ ] Diagnostics JSON valid and complete
- [ ] 15-20 test images covering all categories
- [ ] No sklearn, torch, or deep learning
- [ ] Code is readable with docstrings

## Troubleshooting

**"No images found"**
- Ensure images are in `dataset/input/`
- Check file extensions (.jpg, .png, .bmp supported)

**"Enhanced images look weird"**
- Check treatment order in `main.py` — must apply in sequence
- Review metrics with verbose=True to see what treatments were applied

**"Gallery is blank"**
- Ensure both input and output directories have matching filenames
- Check `gallery/benchmark_grid.png` was created

**"JSON log has null values"**
- Ensure metrics are calculated as floats, not NaN
- Check image was loaded successfully

## References

- **Contrast**: Percentile-based normalization (ISO 19684 standard)
- **Exposure**: Mean/median brightness (standard in digital imaging)
- **Noise**: Median filter residual variance (classic denoising metric)
- **Sharpness**: Laplacian variance (Kurtosis method, Tenengrad operator)
- **Gamma Correction**: Standard nonlinear brightness adjustment (ITU BT.709)
- **Unsharp Masking**: Standard sharpening technique (Photoshop, GIMP)
- **Histogram Equalization**: OpenCV `cv2.equalizeHist()`

## License

Educational use. Part of Mini Project 2 assignment.
