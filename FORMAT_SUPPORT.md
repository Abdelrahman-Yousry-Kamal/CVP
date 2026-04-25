# Image Format Support Documentation

## Overview

The image enhancement pipeline supports a wide range of image formats through intelligent dual-strategy loading:

1. **OpenCV (cv2)** - Fast, standard formats
2. **PIL/Pillow** - Extended format support, modern formats

This approach ensures:
- ✅ Fast loading of standard formats
- ✅ Fallback support for modern/exotic formats
- ✅ Automatic format detection and handling
- ✅ Transparent color space conversion

## Supported Formats

### Standard Formats (OpenCV - native support)

| Extension | Format | Color Modes | Notes |
|-----------|--------|-------------|-------|
| `.jpg`, `.jpeg` | JPEG | RGB/Grayscale | Most common, lossy compression |
| `.png` | PNG | RGBA, RGB, Grayscale | Lossless, with alpha channel |
| `.bmp` | Bitmap | RGB, Grayscale | Uncompressed, large file size |
| `.tiff`, `.tif` | TIFF | RGB, Grayscale, CMYK | High-quality, supports layers |

### Modern Formats (PIL - fallback/extended)

| Extension | Format | Requires | Notes |
|-----------|--------|----------|-------|
| `.webp` | WebP | Pillow 2.8+ | Modern, good compression, lossy/lossless |
| `.avif` | AVIF | Pillow 8.1+ | 2023+ standard, highest compression |
| `.heic`, `.heif` | HEIC/HEIF | Pillow 7.0+ | Apple modern format, iPhone/iPad |
| `.gif` | GIF | Pillow 1.0+ | Animated or static (uses first frame) |

### Additional Formats (Pillow extended)

| Extension | Format | Notes |
|-----------|--------|-------|
| `.ico` | Icon | Windows/Web icon format |
| `.ppm`, `.pgm`, `.pbm` | PPM/PGM/PBM | Portable pixmap formats (P6, P5, P4) |
| `.xcf` | GIMP XCF | GIMP native format (limited support) |
| `.dds` | DirectDraw Surface | DirectX/game asset format |
| `.pcx` | PCX | Legacy DOS era format |
| `.tga` | Targa | Retro graphics format |
| `.eps`, `.pdf` | Vector/PDF | Limited raster support |

## Color Mode Handling

The pipeline automatically detects and converts color modes:

### Input Handling
```
RGBA (8-bit RGBA)     → RGB → BGR (for OpenCV)
LA (Grayscale+Alpha)  → RGB → BGR
L (Grayscale)         → RGB → BGR (converts to 3-channel)
P (Palette/Indexed)   → RGB → BGR
CMYK                  → RGB → BGR
RGB                   → BGR (for OpenCV format)
I/F (Float int/float) → RGB → BGR
```

### Output
All enhanced images are saved as:
- **By default**: JPEG (`.jpg`) - balanced quality and size
- **If specified**: Original format maintained
- **Quality**: 95% for JPEG/WebP/AVIF (minimal quality loss)

## Loading Pipeline

### Step 1: OpenCV (cv2.imread)
```python
img_bgr = cv2.imread(image_path)
if img is not None:
    # Success - use it
    return convert to grayscale
```

Benefits:
- ✓ Fast native C++ implementation
- ✓ Directly returns BGR (our expected format)
- ✓ Handles all common formats

Limitations:
- ✗ Modern formats may not be supported
- ✗ AVIF support depends on OpenCV build configuration

### Step 2: PIL Fallback (Image.open)
```python
img_pil = Image.open(image_path)
# Convert color mode to RGB
img_pil = img_pil.convert('RGB')
# Convert to BGR for OpenCV compatibility
img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
```

Benefits:
- ✓ Supports modern formats (AVIF, WebP, HEIC)
- ✓ Better error handling for corrupted files
- ✓ Handles animated images gracefully

Overhead:
- ✓ Slightly slower (Python-based)
- ✓ But only used when OpenCV fails

## Usage Examples

### Example 1: Load AVIF Image
```python
# File: dataset/input/photo.avif
# Pipeline flow:
# 1. Try cv2.imread() → fails (AVIF not supported)
# 2. Fall back to PIL → success
# 3. Convert AVIF → RGB → BGR
```

### Example 2: Load PNG with Alpha Channel
```python
# File: dataset/input/image.png (RGBA)
# Pipeline flow:
# 1. cv2.imread() → loads but keeps alpha
# 2. This is fine! cv2.cvtColor handles it
# 3. Grayscale conversion works correctly
```

### Example 3: Load Animated GIF
```python
# File: dataset/input/animation.gif (10 frames)
# Pipeline flow:
# 1. cv2.imread() → loads first frame only (via OpenCV default)
# 2. Or PIL loads → explicitly uses first frame via .seek(0)
# 3. Returns BGR image of first frame
```

## Troubleshooting

### Issue: "Error loading image with cv2"
**Cause**: Format not supported by OpenCV build
**Solution**: 
- Compiled PIL will handle it (no action needed)
- Check error message for details
- Fallback to PIL is automatic

### Issue: "Error loading image with PIL"
**Cause**: 
- File is corrupted
- Format is not supported by Pillow version
- Missing optional dependencies
**Solution**:
```bash
# Upgrade Pillow for format support
pip install --upgrade Pillow

# Check installed Pillow formats
python -c "from PIL import Image; print(Image.registered_extensions())"
```

### Issue: "AVIF images not loading"
**Cause**: Pillow version too old or built without AVIF support
**Solution**:
```bash
# Requires Pillow 8.1+ with libaom support
pip install --upgrade Pillow

# Verify AVIF support
python -c "from PIL import Image; print('.avif' in Image.registered_extensions())"
```

### Issue: "Color looks inverted/wrong after loading"
**Cause**: Usually indicates OpenCV BGR vs RGB mismatch
**Solution**:
- This is handled automatically by the pipeline
- All images converted to grayscale for diagnosis
- Pipeline internally uses BGR
- Check your display isn't inverting channels

## Performance Notes

### Loading Speed (by format)
```
Fastest:  JPEG (native OpenCV)      ~1-5ms
Fast:     PNG (native OpenCV)        ~2-10ms
Medium:   TIFF (native OpenCV)       ~5-20ms
Slower:   WebP (via PIL)            ~10-30ms
Slowest:  AVIF (via PIL, complex)   ~20-100ms
          HEIC (via PIL, fallback)  ~20-100ms
```

### File Size (relative to original, for same image)
```
Original PNG:   100% (lossless)
JPEG 95%:       5-15% (lossy, imperceptible)
PNG optimized:  40-70% (lossless)
WebP:           30-50% (lossy comparable)
AVIF:           15-40% (best compression)
```

### Recommendation
- **Input**: Any format (auto-handled)
- **Output**: JPEG for web/sharing
- **Archive**: PNG for lossless preservation
- **Modern**: WebP/AVIF for efficiency

## Advanced: Custom Format Handling

### Save Enhanced Image in Different Format
```python
# Default: saves as JPEG
from utils import save_image
save_image("output.jpg", enhanced_img)

# Custom format:
import cv2
cv2.imwrite("output.avif", enhanced_img)  # If OpenCV supports AVIF
# Or use PIL:
from PIL import Image
img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
Image.fromarray(img_rgb).save("output.avif", quality=95)
```

### Custom Extension List  
```python
from utils import get_image_files

# Get only JPEG files
files = get_image_files("dataset/input", extensions=['.jpg', '.jpeg'])

# Get only modern formats
files = get_image_files("dataset/input", 
                        extensions=['.webp', '.avif', '.heic'])

# Support ALL formats
# (pass None to use default comprehensive list)
files = get_image_files("dataset/input")
```

## Dependencies

### Core
- `opencv-python` ≥4.5 - Fast image I/O
- `Pillow` ≥7.0 - Format fallback

### For Full Format Support
```bash
# Standard install (most formats)
pip install Pillow

# AVIF support (recommended)
pip install Pillow --upgrade

# Specific codec support
pip install "Pillow[webp,heic]"
```

### Version Requirements
| Format | Min Pillow Version | Notes |
|--------|-------------------|-------|
| WebP | 2.8+ | Usually installed |
| AVIF | 8.1+ | Modern, recommended |
| HEIC | 7.0+ | Apple images |
| GIF | 1.0+ | Built-in |

## Quality Settings

All saved images use high-quality settings to preserve enhancement:

```python
# Code: utils.save_image()
if ext in ['.jpg', '.jpeg', '.webp', '.avif']:
    img_pil.save(path_str, quality=95)  # 95% quality
else:
    img_pil.save(path_str)  # Lossless formats, no quality param
```

## Environment Variables

No configuration needed! Format support is automatic.

However, you can override the default OpenCV behavior:

```bash
# Force OpenCV to use certain libraries
export OPENCV_VIDEOIO_DEBUG=1
export OPENCV_LOG_LEVEL=DEBUG

# Then run pipeline
python main.py
```

## References

- **OpenCV formats**: https://docs.opencv.org/master/d4/da8/group__imgcodecs.html
- **Pillow formats**: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
- **AVIF specification**: https://aomediacodec.org/
- **WebP**: https://developers.google.com/speed/webp/
- **HEIC/HEIF**: https://en.wikipedia.org/wiki/High_Efficiency_Image_Container

## Summary

| Need | Format | Action |
|------|--------|--------|
| Input JPEGs | JPEG | ✓ Automatic |
| Input PNGs | PNG | ✓ Automatic |
| Input AVIFs | AVIF | ✓ Automatic (via PIL) |
| Input any format | Mixed | ✓ Automatic (dual-strategy) |
| Output enhanced | Any | ✓ Specify in filename or code |
| Highest compression | AVIF | ✓ Use `.avif` extension |
| Web-compatible | WebP | ✓ Use `.webp` extension |
| Lossless archive | PNG | ✓ Use `.png` extension |

**Bottom line**: Just drop images in `dataset/input/` - all formats are handled automatically! 📁✨
