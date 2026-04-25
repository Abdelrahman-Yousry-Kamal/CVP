"""
QUICK START GUIDE

This file provides copy-paste commands for getting started with the project.

STEP 1: Install dependencies
    pip install numpy opencv-python scipy pillow matplotlib

STEP 2: Create test dataset (optional but recommended)
    Option A - Auto-generate degraded images from clean base images:
        1. Place 3-4 clean reference images in: dataset/input_base/
           (supports: jpg, jpeg, png, bmp, tiff, gif, webp, avif, heic)
        2. Run: python dataset_generator.py
        3. This will create 16-19 test images in dataset/input/
    
    Option B - Manually add images:
        1. Place 15-20 images directly in: dataset/input/
        2. Supported formats:
           • Standard: jpg, jpeg, png, bmp, tiff, tif, gif
           • Modern: webp, avif, heic, heif
           • Other: ico, ppm, pgm, pbm, xcf, dds

STEP 3: Run the pipeline
    python main.py

    Output:
    • dataset/output/        → Enhanced images
    • logs/diagnostics.json  → Per-image metrics and treatments
    • gallery/benchmark_grid.png → Visual before/after/histogram

STEP 4: Inspect results
    • Open gallery/benchmark_grid.png in image viewer
    • Read logs/diagnostics.json for detailed metrics
    • Compare original vs enhanced in dataset/output/

========================================================================================

PROJECT STRUCTURE AT A GLANCE

diagnosis.py
    └─ diagnose(gray_img) → {contrast, exposure, noise, sharpness}

controller.py  
    └─ prescribe(metrics) → [treatment_1, treatment_2, ...]
    └─ diagnose_and_prescribe(metrics) → (pipeline, report)

treatment.py
    └─ TREATMENT_MAP = {"gamma_correction_brighten": fn, ...}
    └─ Functions: gamma_correction, median_filter, gaussian_filter, 
                  contrast_stretching, histogram_equalization, 
                  unsharp_mask, laplacian_sharpen

utils.py
    └─ load_image(path) → (img_bgr, img_gray)
    └─ save_image(path, img) → bool
    └─ log_diagnostics(data, log_path)
    └─ get_image_files(folder, extensions)

gallery_generator.py
    └─ build_gallery_advanced(input_dir, output_dir, save_path, metrics_list)

main.py
    └─ process_image(input_path, output_path) → (metrics, treatments, success)
    └─ main() → Orchestrates everything

dataset_generator.py
    └─ generate_dataset(base_images_dir, output_dir) → count

========================================================================================

PARAMETER REFERENCE (from controller.py)

CONTRAST_LOW = 50               # Contrast < 50 → needs enhancement
CONTRAST_HIGH = 200             # Contrast > 200 → already very high
EXPOSURE_DARK = 60              # Mean < 60 → underexposed
EXPOSURE_BRIGHT = 190           # Mean > 190 → overexposed
NOISE_LOW = 20                  # Variance < 20 → clean
NOISE_HIGH = 80                 # Variance > 80 → heavy salt-and-pepper noise
SHARPNESS_LOW = 100             # Variance < 100 → blurry
SHARPNESS_MEDIUM = 500          # Variance < 500 → slightly soft

========================================================================================

TREATMENT ORDER (Cardinal Rule - 30% of grade!)

1. EXPOSURE CORRECTION (if needed)
   → Γ-brighten (if exposure < 60)
   → Γ-darken (if exposure > 190)

2. NOISE REMOVAL (if needed)
   → Median filter (if noise > 80, heavy salt-and-pepper)
   → Gaussian filter (if 20 < noise ≤ 80, mild Gaussian)

3. CONTRAST ENHANCEMENT (if needed and contrast < 50)
   → Contrast stretching (safer on noisy images)
   → Histogram equalization (more aggressive on clean images)

4. SHARPENING (only if safe: noise < 80)
   → Unsharp mask (if sharpness < 100)
   → Laplacian sharpen (if 100 < sharpness < 500)

========================================================================================

EXPECTED OUTPUTS

For a folder with 3 images:

dataset/output/
├── image_01.jpg  ← Enhanced
├── image_02.jpg  ← Enhanced
└── image_03.jpg  ← Enhanced

logs/diagnostics.json:
[
  {
    "image_filename": "image_01.jpg",
    "metrics": {
      "contrast": 85.2,
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
  },
  ...
]

gallery/benchmark_grid.png:
    3-column grid (Original | Histogram | Enhanced) for each image
    Height: 4" per image, Width: 18" total

========================================================================================

TESTING CHECKLIST

Testing single image manually:
    from diagnosis import diagnose
    from controller import prescribe, diagnose_and_prescribe
    import cv2
    
    img = cv2.imread("path/to/image.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    metrics = diagnose(gray)
    print(metrics)
    
    treatments, report = diagnose_and_prescribe(metrics)
    print(f"Treatments: {treatments}")
    print(f"Report: {report}")

========================================================================================

COMMON ISSUES

Issue: "Image loads but enhancement has no effect"
Solution: 
  • Check if image is "perfect" (all metrics in acceptable range)
  • Add a tiny bit of noise to test: add_gaussian_noise(img, std=10)
  • Manually darken/brighten to trigger exposure correction

Issue: "Sharpened image looks worse, has noise artifacts"
Solution:
  • Noise is being amplified by sharpening
  • Ensure controller is applying noise removal BEFORE sharpening
  • Check NOISE_HIGH threshold (default 80)

Issue: "Color looks destroyed after equalization"
Solution:
  • Equalization must preserve color channels
  • Code uses YCrCb color space (correct)
  • Check that cv2.imwrite is saving as BGR, not RGB

Issue: "Gallery image is blank or tiny"
Solution:
  • Ensure dataset/input/ and dataset/output/ have same filenames
  • Check that gallery save path parent directory exists
  • Try matplotlib.pyplot.show() before saving for debugging

========================================================================================

PERFORMANCE TIPS

• Smaller images process faster (resize to 800x600 if testing)
• Large kernels in blur/median slow down processing
• For 1000+ images, consider parallel processing
• Numpy vectorization > scipy loops

========================================================================================

SUPPORTED IMAGE FORMATS

The pipeline supports a comprehensive range of image formats through dual-strategy loading:

PRIMARY FORMATS (OpenCV):
  • JPEG (.jpg, .jpeg)      - Standard, fast loading
  • PNG (.png)              - Lossless with alpha support
  • BMP (.bmp)              - Uncompressed bitmap
  • TIFF (.tiff, .tif)      - High-quality professional format

MODERN FORMATS (PIL Fallback):
  • WebP (.webp)            - Modern compression format
  • AVIF (.avif)            - Highest compression ratio (requires Pillow 8.1+)
  • HEIC/HEIF (.heic, .heif) - Apple modern format
  • GIF (.gif)              - Animated (first frame used) or static

OTHER FORMATS:
  • ICO (.ico)              - Icon format
  • PPM/PGM/PBM (.ppm, .pgm, .pbm) - Portable pixmap formats
  • GIMP (.xcf)             - GIMP native format (limited support)
  • DirectDraw (.dds)       - DirectX format

LOADING STRATEGY:
1. Try OpenCV (cv2.imread) - fast, standard
2. Fallback to PIL (Image.open) - better compatibility
3. Auto-convert color modes (RGBA→RGB, L→RGB, etc.)
4. Handle animated formats (GIF) - uses first frame

SAVING FORMAT:
• Enhanced images always saved as JPEG by default (.jpg)
• To save in different format, modify output filename extension
• PIL handles modern formats (AVIF, WebP) automatically

EXAMPLE - Process AVIF image:
    Place: dataset/input/photo01.avif
    Output: dataset/output/photo01.jpg (enhanced JPEG)
    Or specify: dataset/output/photo01.avif (if using utils.save_image(..., format_specific=True))

========================================================================================

DEPENDENCIES VERSION INFO

numpy          1.20+  (pixel math, percentiles, var)
opencv-python  4.5+   (image I/O, filters, Laplacian)
scipy          1.5+   (median_filter from ndimage)
pillow         7.0+   (format compatibility fallback)
matplotlib     3.3+   (gallery visualization)

========================================================================================
"""
