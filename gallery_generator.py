"""
Gallery Generator Module
Builds the visual benchmark grid for grading.
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import numpy as np


def build_gallery(input_dir, output_dir, save_path, title="Image Enhancement Benchmark Gallery"):
    """
    Build a comprehensive 3-column benchmark gallery showing original, 
    histogram, and enhanced versions of each image.
    
    Layout:
    - Column 1: Original image
    - Column 2: Histogram visualization
    - Column 3: Enhanced image
    
    Args:
        input_dir: Path to directory with original images
        output_dir: Path to directory with enhanced images
        save_path: Path where benchmark grid PNG will be saved
        title: Title for the gallery (default="Image Enhancement Benchmark Gallery")
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get list of image files
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Find all image files (case-insensitive) - support all common formats
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
                            '.gif', '.webp', '.avif', '.heic', '.heif'}
        image_files = sorted([
            f for f in input_path.glob('*')
            if f.suffix.lower() in supported_formats
        ])
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return False
        
        num_images = len(image_files)
        
        # Create figure with 3 columns and num_images rows
        # Height: 4 inches per row, Width: 18 inches (6 per column)
        fig, axes = plt.subplots(num_images, 3, figsize=(18, 4 * num_images))
        
        # Handle single image case (axes is 1D array instead of 2D)
        if num_images == 1:
            axes = axes.reshape(1, -1)
        
        # Add title to the figure
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        
        # Process each image
        row_idx = 0  # Track actual row being drawn
        for input_file in image_files:
            filename = input_file.name
            output_file = output_path / filename
            
            # Load original image
            original_bgr = cv2.imread(str(input_file))
            if original_bgr is None:
                print(f"⚠ Warning: Could not load original {input_file}")
                continue
            
            # Convert BGR to RGB for display
            original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
            
            # Load enhanced image
            enhanced_rgb = None
            if output_file.exists():
                enhanced_bgr = cv2.imread(str(output_file))
                if enhanced_bgr is not None:
                    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                else:
                    print(f"⚠ Warning: Could not load enhanced {filename}, using original")
                    enhanced_rgb = original_rgb.copy()
            else:
                # If enhanced version doesn't exist
                print(f"⚠ Warning: Enhanced version not found for {filename}, using original")
                enhanced_rgb = original_rgb.copy()
            
            # Convert to grayscale for histogram
            gray_original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
            
            # COLUMN 1: Original Image
            axes[row_idx, 0].imshow(original_rgb)
            axes[row_idx, 0].set_title(f"Original: {filename}", fontsize=10, fontweight='bold')
            axes[row_idx, 0].axis('off')
            
            # COLUMN 2: Histogram
            axes[row_idx, 1].hist(gray_original.ravel(), bins=256, color='gray', alpha=0.7, edgecolor='black')
            axes[row_idx, 1].set_title("Histogram (Grayscale)", fontsize=10, fontweight='bold')
            axes[row_idx, 1].set_xlabel("Pixel Intensity")
            axes[row_idx, 1].set_ylabel("Frequency")
            axes[row_idx, 1].grid(alpha=0.3)
            # Set x-axis limits to [0, 255] for consistency
            axes[row_idx, 1].set_xlim([0, 255])
            
            # COLUMN 3: Enhanced Image
            axes[row_idx, 2].imshow(enhanced_rgb)
            if enhanced_rgb is original_rgb:
                axes[row_idx, 2].set_title("Enhanced (not processed)", fontsize=10, fontweight='bold')
            else:
                axes[row_idx, 2].set_title("Enhanced", fontsize=10, fontweight='bold')
            axes[row_idx, 2].axis('off')
            
            row_idx += 1  # Move to next row
        
        # If no images were successfully loaded, close figure and return false
        if row_idx == 0:
            print("Error: No images were successfully loaded for gallery")
            plt.close(fig)
            return False
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Ensure output directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save figure
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Gallery saved: {save_path} ({row_idx} images)")
        
        # Close figure to free memory
        plt.close(fig)
        
        return True
    
    except Exception as e:
        print(f"Error building gallery: {e}")
        return False


def build_gallery_advanced(input_dir, output_dir, save_path, metrics_list=None, 
                          title="Image Enhancement Benchmark Gallery"):
    """
    Build an advanced gallery with optional metric annotations.
    
    This version can display metric values from the diagnostic report
    alongside each image for more detailed analysis.
    
    Args:
        input_dir: Path to directory with original images
        output_dir: Path to directory with enhanced images
        save_path: Path where benchmark grid PNG will be saved
        metrics_list: Optional list of metric dictionaries (one per image)
                     Format: [{"contrast": ..., "exposure": ..., ...}, ...]
        title: Title for the gallery
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get list of image files
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Find all image files - support all common formats
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
                            '.gif', '.webp', '.avif', '.heic', '.heif'}
        image_files = sorted([
            f for f in input_path.glob('*')
            if f.suffix.lower() in supported_formats
        ])
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return False
        
        num_images = len(image_files)
        
        # Create figure
        fig, axes = plt.subplots(num_images, 3, figsize=(18, 5 * num_images))
        
        if num_images == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        
        # Process each image
        for i, input_file in enumerate(image_files):
            filename = input_file.name
            output_file = output_path / filename
            
            # Load images
            original_bgr = cv2.imread(str(input_file))
            if original_bgr is None:
                continue
            
            original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
            gray_original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
            
            if output_file.exists():
                enhanced_bgr = cv2.imread(str(output_file))
                enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            else:
                enhanced_rgb = original_rgb.copy()
            
            # Build subtitle with metrics if available
            subtitle1 = f"Original: {filename}"
            if metrics_list and i < len(metrics_list):
                m = metrics_list[i]
                if isinstance(m, dict) and 'metrics' in m:
                    metrics = m['metrics']
                    subtitle1 += f"\n(C:{metrics['contrast']:.1f}, E:{metrics['exposure']:.1f})"
            
            # Column 1: Original
            axes[i, 0].imshow(original_rgb)
            axes[i, 0].set_title(subtitle1, fontsize=9, fontweight='bold')
            axes[i, 0].axis('off')
            
            # Column 2: Histogram
            axes[i, 1].hist(gray_original.ravel(), bins=256, color='gray', alpha=0.7, edgecolor='black')
            axes[i, 1].set_title("Histogram", fontsize=9, fontweight='bold')
            axes[i, 1].set_xlabel("Pixel Intensity", fontsize=8)
            axes[i, 1].set_ylabel("Count", fontsize=8)
            axes[i, 1].set_xlim([0, 255])
            axes[i, 1].tick_params(labelsize=8)
            axes[i, 1].grid(alpha=0.3)
            
            # Column 3: Enhanced
            subtitle3 = "Enhanced"
            if metrics_list and i < len(metrics_list):
                m = metrics_list[i]
                if isinstance(m, dict) and 'treatments_applied' in m:
                    treatments = m['treatments_applied']
                    if treatments:
                        subtitle3 += f"\n({len(treatments)} treatments)"
                    else:
                        subtitle3 += "\n(no adjustments)"
            
            axes[i, 2].imshow(enhanced_rgb)
            axes[i, 2].set_title(subtitle3, fontsize=9, fontweight='bold')
            axes[i, 2].axis('off')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Advanced gallery saved to {save_path}")
        
        plt.close(fig)
        
        return True
    
    except Exception as e:
        print(f"Error building advanced gallery: {e}")
        return False


def build_side_by_side(before_image, after_image, metrics=None, save_path=None):
    """
    Build a simpler side-by-side comparison of a single image pair.
    
    Useful for individual image inspection and testing.
    
    Args:
        before_image: Original image (numpy array, BGR)
        after_image: Enhanced image (numpy array, BGR)
        metrics: Optional dictionary of metrics for display
        save_path: Optional path to save the comparison (if None, returns fig)
    
    Returns:
        Figure object if save_path is None, otherwise True/False for save success
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        before_rgb = cv2.cvtColor(before_image, cv2.COLOR_BGR2RGB)
        after_rgb = cv2.cvtColor(after_image, cv2.COLOR_BGR2RGB)
        
        axes[0].imshow(before_rgb)
        axes[0].set_title("Before Enhancement", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(after_rgb)
        axes[1].set_title("After Enhancement", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        if metrics:
            # Add metrics as figure text
            metrics_text = "Metrics:\n"
            for key, value in metrics.items():
                if isinstance(value, float):
                    metrics_text += f"{key}: {value:.2f}\n"
                else:
                    metrics_text += f"{key}: {value}\n"
            fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=9, family='monospace')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return True
        else:
            return fig
    
    except Exception as e:
        print(f"Error building side-by-side comparison: {e}")
        return False
