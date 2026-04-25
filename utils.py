"""
Utilities Module
Helper functions for:
- Image loading with fallback 
- Image saving
- JSON logging of diagnostic reports
- Folder I/O pipeline management
"""
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
# IMAGE LOADING
def load_image_cv2(image_path):
    """
    Load image using OpenCV.
    
    Args:
        image_path: Path to image file (string or Path object)
    
    Returns:
        Tuple (img_bgr, img_gray) or (None, None) if load fails
    """
    try:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return None, None
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return img_bgr, img_gray
    except Exception as e:
        print(f"Error loading image with cv2: {image_path} - {e}")
        return None, None


def load_image_pil(image_path):
    """
    Load image using PIL as fallback (better format compatibility).
    
    Supports: JPEG, PNG, GIF, TIFF, BMP, WEBP, AVIF, ICO, PPM, and others.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Tuple (img_bgr, img_gray) in OpenCV format or (None, None) if fails
    """
    try:
        # Load with PIL
        img_pil = Image.open(image_path)
        
        # Handle animated formats (GIF) - take first frame
        if hasattr(img_pil, 'n_frames') and img_pil.n_frames > 1:
            img_pil.seek(0)
        
        # Handle various color modes
        if img_pil.mode in ('RGBA', 'LA', 'P'):
            # Convert RGBA/LA/palette images to RGB
            img_pil = img_pil.convert('RGB')
        elif img_pil.mode == 'L':
            # Grayscale: convert to RGB for consistency
            img_pil = img_pil.convert('RGB')
        elif img_pil.mode not in ('RGB', 'BGR'):
            # Any other mode: convert to RGB
            img_pil = img_pil.convert('RGB')
        
        # Convert to OpenCV format (BGR)
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        return img_bgr, img_gray
    except Exception as e:
        print(f"Error loading image with PIL: {image_path} - {e}")
        return None, None


def load_image(image_path):
    """
    Load image with fallback strategy.
    
    1. Try OpenCV first (fast, standard)
    2. Fall back to PIL if OpenCV fails (better format support)
    
    Args:
        image_path: Path to image file
    
    Returns:
        Tuple (img_bgr, img_gray) or (None, None) if both fail
    """
    img_bgr, img_gray = load_image_cv2(image_path)
    if img_bgr is None:
        img_bgr, img_gray = load_image_pil(image_path)
    return img_bgr, img_gray
# IMAGE SAVING
def save_image(image_path, img, format_specific=False):
    """
    Save image using OpenCV (with PIL fallback for unsupported formats).
    
    Supports: JPEG, PNG, GIF, TIFF, BMP, WEBP, AVIF, ICO, PPM, and others.
    
    Args:
        image_path: Output path (string or Path object)
        img: Image to save (BGR for color, grayscale for single channel)
        format_specific: If True, use PIL for better format handling. Default=False.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create parent directory if it doesn't exist
        Path(image_path).parent.mkdir(parents=True, exist_ok=True)
        path_str = str(image_path)
        
        # Determine file extension
        ext = Path(image_path).suffix.lower()
        
        # For modern formats (AVIF, WEBP), prefer PIL
        if format_specific or ext in ['.avif', '.webp', '.heic', '.heif']:
            try:
                # Convert BGR to RGB for PIL
                if len(img.shape) == 3:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                else:
                    # Grayscale
                    img_pil = Image.fromarray(img)
                
                # Set quality for JPEG/WEBP/AVIF
                if ext in ['.jpg', '.jpeg', '.webp', '.avif']:
                    img_pil.save(path_str, quality=95)
                else:
                    img_pil.save(path_str)
                return True
            except Exception as e:
                print(f"Warning: PIL save failed for {image_path}: {e}. Trying OpenCV...")
                # Fall through to OpenCV
        
        # Use OpenCV for standard formats
        success = cv2.imwrite(path_str, img)
        if not success:
            print(f"Error: cv2.imwrite failed to save {image_path}")
            return False
        return True
    except Exception as e:
        print(f"Error saving image: {image_path} - {e}")
        return False
# JSON LOGGING
def log_diagnostics(log_data, log_path="logs/diagnostics.json"):
    """
    Append diagnostic data for one image to the JSON log file.
    
    Log format:
    {
        "image_filename": "",
        "metrics": {
            "contrast": float,
            "exposure": float,
            "noise": float,
            "sharpness": float
        },
        "metric_classes": {
            "contrast_status": str,
            "exposure_status": str,
            "noise_status": str,
            "sharpness_status": str
        },
        "treatments_applied": [str, ...],
        "treatment_count": int
    }
    
    Args:
        log_data: Dictionary with diagnostic information
        log_path: Path to JSON log file. Default="logs/diagnostics.json"
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create parent directory if needed
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing log or create new list
        if Path(log_path).exists() and Path(log_path).stat().st_size > 0:
            try:
                with open(log_path, 'r') as f:
                    log_list = json.load(f)
            except json.JSONDecodeError:
                log_list = []
        else:
            log_list = []
        
        # Append new entry
        log_list.append(log_data)
        
        # Write back to file
        with open(log_path, 'w') as f:
            json.dump(log_list, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error logging diagnostics: {e}")
        return False
def create_log_entry(filename, metrics, diagnostic_report):
    """
    Create a single log entry from processed image data.
    
    Args:
        filename: Original image filename
        metrics: Dictionary from diagnose() with raw metric values
        diagnostic_report: Dictionary from diagnose_and_prescribe() with classifications
    
    Returns:
        Dictionary formatted for JSON logging
    """
    return {
        "image_filename": filename,
        "metrics": {
            "contrast": float(metrics["contrast"]),
            "exposure": float(metrics["exposure"]),
            "noise": float(metrics["noise"]),
            "sharpness": float(metrics["sharpness"])
        },
        "metric_classes": {
            "contrast_status": diagnostic_report["contrast_status"],
            "exposure_status": diagnostic_report["exposure_status"],
            "noise_status": diagnostic_report["noise_status"],
            "sharpness_status": diagnostic_report["sharpness_status"]
        },
        "treatments_applied": diagnostic_report["treatments"],
        "treatment_count": diagnostic_report["treatment_count"]
    }
# FOLDER MANAGEMENT
def get_image_files(folder_path, extensions=None):
    """
    Get list of image files from folder.
    
    Supports common and modern formats: JPEG, PNG, BMP, TIFF, GIF, WEBP, AVIF, ICO, PPM, etc.
    
    Args:
        folder_path: Path to folder
        extensions: List of file extensions to look for.
                   Default includes JPEG, PNG, BMP, TIFF, GIF, WEBP, AVIF, ICO, PPM
    
    Returns:
        Sorted list of Path objects for image files
    """
    if extensions is None:
        # Comprehensive list of supported formats
        extensions = [
            # Standard formats
            '.jpg', '.jpeg',        # JPEG
            '.png',                 # PNG
            '.bmp',                 # Bitmap
            '.tiff', '.tif',        # TIFF
            '.gif',                 # GIF (animated or static)
            
            # Modern formats
            '.webp',                # WebP (modern)
            '.avif',                # AVIF (modern, highest compression)
            '.heic', '.heif',       # HEIC/HEIF (Apple modern format)
            
            # Other formats
            '.ico',                 # Icon
            '.ppm', '.pgm', '.pbm', # PPM formats
            '.xcf',                 # GIMP
            '.dds',                 # DirectDraw Surface
        ]
    
    folder = Path(folder_path)
    
    # Find all files matching extensions (case-insensitive)
    files = []
    for ext in extensions:
        files.extend(folder.glob(f'*{ext}'))
        files.extend(folder.glob(f'*{ext.upper()}'))
    
    return sorted(set(files))  # Remove duplicates and sort


def ensure_directories():
    """
    Ensure all required output directories exist.
    
    Returns:
        Dictionary with paths to required directories
    """
    dirs = {
        'input': Path('dataset/input'),
        'output': Path('dataset/output'),
        'gallery': Path('gallery'),
        'logs': Path('logs')
    }
    
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    
    return dirs
# METADATA HELPERS
def get_image_info(img):
    """
    Get basic information about an image.
    
    Args:
        img: OpenCV image (BGR or grayscale)
    
    Returns:
        Dictionary with image properties
    """
    if img is None:
        return None
    
    info = {
        "shape": img.shape,
        "dtype": str(img.dtype),
        "min_pixel": float(np.min(img)),
        "max_pixel": float(np.max(img)),
        "mean_pixel": float(np.mean(img)),
        "std_pixel": float(np.std(img))
    }
    
    if len(img.shape) == 3:
        info["channels"] = img.shape[2]
        info["height"] = img.shape[0]
        info["width"] = img.shape[1]
    else:
        info["channels"] = 1
        info["height"] = img.shape[0]
        info["width"] = img.shape[1]
    
    return info
