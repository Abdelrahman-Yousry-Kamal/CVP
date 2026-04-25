"""
Dataset Generator
Helper script to programmatically generate test images with controlled degradations.

This is NOT submitted as part of the project — it's just for building your dataset.
Generate degraded versions from 3-4 base images to ensure you stress-test every code path.

Coverage Strategy (15-20 images total):
- 3 Perfect/reference images (clean photos)
- 3 Dark/underexposed images
- 3 Bright/overexposed images
- 3 Salt-and-pepper noisy images
- 3 Gaussian noisy images
- 2 Blurry images
- 2 Multiple defects (dark+noisy, blurry+low contrast)
"""

import cv2
import numpy as np
from pathlib import Path


def load_image_flexible(image_path):
    """
    Load image with robust format handling (OpenCV, then PIL fallback).
    Returns only BGR image for processing (not grayscale).
    """
    try:
        # Try OpenCV first
        img = cv2.imread(str(image_path))
        if img is not None:
            return img
    except Exception:
        pass
    
    # Fallback to PIL for modern formats
    try:
        from PIL import Image
        img_pil = Image.open(image_path)
        if img_pil.mode in ('RGBA', 'LA', 'P'):
            img_pil = img_pil.convert('RGB')
        elif img_pil.mode not in ('RGB', 'BGR'):
            img_pil = img_pil.convert('RGB')
        img_array = np.array(img_pil)
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Failed to load {image_path}: {e}")
        return None


def add_salt_pepper_noise(img, amount=0.04):
    """
    Add salt-and-pepper noise to image.
    
    Args:
        img: Input image (BGR or grayscale)
        amount: Fraction of pixels to corrupt (0.0-1.0). Default=0.04 (4%)
    
    Returns:
        Noisy image
    """
    noisy = img.copy()
    n_pixels = int(amount * img.size)
    
    # Add salt (white pixels)
    coords = [np.random.randint(0, i - 1, n_pixels) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    
    # Add pepper (black pixels)
    coords = [np.random.randint(0, i - 1, n_pixels) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    
    return noisy


def add_gaussian_noise(img, mean=0, std=25):
    """
    Add Gaussian/random noise to image.
    
    Args:
        img: Input image (BGR or grayscale)
        mean: Mean of Gaussian noise. Default=0
        std: Standard deviation. Default=25 (mild-moderate noise)
    
    Returns:
        Noisy image (clipped to [0, 255])
    """
    noise = np.random.normal(mean, std, img.shape)
    noisy = img.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def darken_image(img, factor=0.25):
    """
    Darken image by multiplying pixel values.
    
    Args:
        img: Input image
        factor: Multiplication factor (0.0-1.0). Default=0.25 (75% darker)
    
    Returns:
        Darkened image
    """
    darkened = img.astype(np.float64) * factor
    return np.clip(darkened, 0, 255).astype(np.uint8)


def brighten_image(img, factor=2.0):
    """
    Brighten image by multiplying pixel values.
    
    Args:
        img: Input image
        factor: Multiplication factor (>1.0). Default=2.0 (2x brighter)
    
    Returns:
        Brightened image (clipped to [0, 255])
    """
    brightened = img.astype(np.float64) * factor
    return np.clip(brightened, 0, 255).astype(np.uint8)


def blur_image(img, ksize=25):
    """
    Blur image with Gaussian blur to simulate focus problems.
    
    Args:
        img: Input image
        ksize: Kernel size (must be odd). Default=25. Larger = more blur
    
    Returns:
        Blurred image
    """
    # Ensure ksize is odd
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def lower_contrast(img, factor=0.3):
    """
    Lower contrast by mapping pixel values toward the middle (128).
    
    Simulates a washed-out, flat image.
    
    Args:
        img: Input image
        factor: How much to compress toward middle (0.0-1.0)
    
    Returns:
        Low-contrast image
    """
    middle = 128
    low_contrast = (img.astype(np.float64) - middle) * (1 - factor) + middle
    return np.clip(low_contrast, 0, 255).astype(np.uint8)


def generate_dataset(base_images_dir, output_dir, num_copies=1):
    """
    Generate complete test dataset from base images.
    
    This function loads a few base images and creates degraded versions
    to hit all the test categories.
    
    Args:
        base_images_dir: Directory with 3-4 base clean images to degrade
        output_dir: Directory to save generated dataset
        num_copies: Number of each degradation type to create (default=1)
    
    Returns:
        Number of images generated
    """
    base_path = Path(base_images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find base images (support all common formats)
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', 
                        '.gif', '.webp', '.avif', '.heic', '.heif'}
    base_files = sorted([
        f for f in base_path.glob('*')
        if f.suffix.lower() in supported_formats
    ])
    
    if not base_files:
        print(f"No base images found in {base_images_dir}")
        return 0
    
    print(f"Found {len(base_files)} base images. Generating degraded versions...")
    print()
    
    count = 0
    base_idx = 0
    
    # Category 1: Perfect/reference images (just copy them)
    for i in range(3):
        base_img = load_image_flexible(base_files[base_idx % len(base_files)])
        if base_img is None:
            base_idx += 1
            continue
        output_file = output_path / f"001_perfect_{i+1:02d}.jpg"
        cv2.imwrite(str(output_file), base_img)
        print(f"✓ {output_file.name}: Perfect/reference")
        count += 1
        base_idx += 1
    
    # Category 2: Underexposed (dark)
    for i in range(3):
        base_img = load_image_flexible(base_files[base_idx % len(base_files)])
        if base_img is None:
            base_idx += 1
            continue
        dark_img = darken_image(base_img, factor=0.2)
        output_file = output_path / f"002_underexposed_{i+1:02d}.jpg"
        cv2.imwrite(str(output_file), dark_img)
        print(f"✓ {output_file.name}: Underexposed (20% brightness)")
        count += 1
        base_idx += 1
    
    # Category 3: Overexposed (bright)
    for i in range(3):
        base_img = load_image_flexible(base_files[base_idx % len(base_files)])
        if base_img is None:
            base_idx += 1
            continue
        bright_img = brighten_image(base_img, factor=2.0)
        output_file = output_path / f"003_overexposed_{i+1:02d}.jpg"
        cv2.imwrite(str(output_file), bright_img)
        print(f"✓ {output_file.name}: Overexposed (2x brightness)")
        count += 1
        base_idx += 1
    
    # Category 4: Salt-and-pepper noise
    for i in range(3):
        base_img = load_image_flexible(base_files[base_idx % len(base_files)])
        if base_img is None:
            base_idx += 1
            continue
        noisy_img = add_salt_pepper_noise(base_img, amount=0.04)
        output_file = output_path / f"004_salt_pepper_{i+1:02d}.jpg"
        cv2.imwrite(str(output_file), noisy_img)
        print(f"✓ {output_file.name}: Salt-and-pepper noise (4%)")
        count += 1
        base_idx += 1
    
    # Category 5: Gaussian noise
    for i in range(3):
        base_img = load_image_flexible(base_files[base_idx % len(base_files)])
        if base_img is None:
            base_idx += 1
            continue
        noisy_img = add_gaussian_noise(base_img, std=30)
        output_file = output_path / f"005_gaussian_noise_{i+1:02d}.jpg"
        cv2.imwrite(str(output_file), noisy_img)
        print(f"✓ {output_file.name}: Gaussian noise (std=30)")
        count += 1
        base_idx += 1
    
    # Category 6: Blurry
    for i in range(2):
        base_img = load_image_flexible(base_files[base_idx % len(base_files)])
        if base_img is None:
            base_idx += 1
            continue
        blurry_img = blur_image(base_img, ksize=25)
        output_file = output_path / f"006_blurry_{i+1:02d}.jpg"
        cv2.imwrite(str(output_file), blurry_img)
        print(f"✓ {output_file.name}: Blurry (25x25 Gaussian)")
        count += 1
        base_idx += 1
    
    # Category 7: Multiple defects - Dark + Noisy
    for i in range(1):
        base_img = load_image_flexible(base_files[base_idx % len(base_files)])
        if base_img is None:
            base_idx += 1
            continue
        degraded = darken_image(base_img, factor=0.3)
        degraded = add_gaussian_noise(degraded, std=35)
        output_file = output_path / f"007_dark_noisy_{i+1:02d}.jpg"
        cv2.imwrite(str(output_file), degraded)
        print(f"✓ {output_file.name}: Dark + Gaussian noise")
        count += 1
        base_idx += 1
    
    # Category 8: Multiple defects - Blurry + Low Contrast
    for i in range(1):
        base_img = load_image_flexible(base_files[base_idx % len(base_files)])
        if base_img is None:
            base_idx += 1
            continue
        degraded = blur_image(base_img, ksize=15)
        degraded = lower_contrast(degraded, factor=0.4)
        output_file = output_path / f"008_blurry_lowcontrast_{i+1:02d}.jpg"
        cv2.imwrite(str(output_file), degraded)
        print(f"✓ {output_file.name}: Blurry + Low contrast")
        count += 1
        base_idx += 1
    
    print()
    print(f"Generated {count} test images in {output_dir}")
    print()
    print("Dataset breakdown:")
    print("  • 3 Perfect/reference images")
    print("  • 3 Underexposed images")
    print("  • 3 Overexposed images")
    print("  • 3 Salt-and-pepper noisy images")
    print("  • 3 Gaussian noisy images")
    print("  • 2 Blurry images")
    print("  • 2 Multiple-defect images (dark+noisy, blurry+low contrast)")
    print(f"  • Total: {count} images")
    
    return count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate test dataset from base images"
    )
    parser.add_argument(
        "--base-dir",
        default="dataset/input_base",
        help="Directory with base clean images"
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/input",
        help="Output directory for generated dataset"
    )
    
    args = parser.parse_args()
    
    generate_dataset(args.base_dir, args.output_dir)
