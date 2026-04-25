"""
Main Entry Point
Runs the complete image enhancement pipeline on all images in dataset/input/.

Pipeline Flow:
1. Load image from dataset/input/
2. Convert to grayscale for diagnosis
3. Phase 1: Diagnose (calculate metrics)
4. Phase 2: Prescribe (determine treatments)
5. Phase 3: Treat (apply enhancements in correct order)
6. Save enhanced image to dataset/output/
7. Log diagnostics to logs/diagnostics.json

After processing all images:
8. Generate benchmark gallery at gallery/benchmark_grid.png
"""

import cv2
from pathlib import Path
from diagnosis import diagnose
from controller import prescribe, diagnose_and_prescribe
from treatment import TREATMENT_MAP
from utils import (
    load_image, save_image, log_diagnostics, create_log_entry,
    get_image_files, ensure_directories, get_image_info
)
from gallery_generator import build_gallery, build_gallery_advanced


def process_image(input_path, output_path, verbose=True):
    """
    Process a single image through the full pipeline.
    
    Steps:
    1. Load image
    2. Convert to grayscale
    3. Diagnose (Phase 1)
    4. Prescribe treatments (Phase 2)
    5. Apply treatments (Phase 3)
    6. Save result
    
    Args:
        input_path: Path to input image
        output_path: Path to save enhanced image
        verbose: Print status messages if True
    
    Returns:
        Tuple (metrics, treatments, success)
        - metrics: Dictionary of diagnostic values
        - treatments: List of applied treatment names
        - success: Boolean indicating success
    """
    # Load image
    img_bgr, img_gray = load_image(input_path)
    if img_bgr is None:
        if verbose:
            print(f"✗ Failed to load: {input_path}")
        return None, None, False
    
    # Phase 1: Diagnose
    metrics = diagnose(img_gray)
    
    # Phase 2: Prescribe
    treatments, report = diagnose_and_prescribe(metrics)
    
    # Phase 3: Treat (apply enhancements in correct order)
    result = img_bgr.copy()
    for treatment_name in treatments:
        if treatment_name in TREATMENT_MAP:
            treatment_func = TREATMENT_MAP[treatment_name]
            result = treatment_func(result)
    
    # Save enhanced image
    if not save_image(output_path, result):
        if verbose:
            print(f"✗ Failed to save: {output_path}")
        return metrics, treatments, False
    
    if verbose:
        status = f"({', '.join(treatments)})" if treatments else "(no adjustments)"
        print(f"✓ {Path(input_path).name:40} → {', '.join([t.replace('_', ' ').title() for t in treatments[:2]])} {status if len(treatments) > 2 else ''}")
    
    return metrics, treatments, True


def main(input_dir="dataset/input", output_dir="dataset/output", 
         log_file="logs/diagnostics.json", gallery_path="gallery/benchmark_grid.png",
         verbose=True):
    """
    Main pipeline: Process all images and generate gallery.
    
    Args:
        input_dir: Directory with input images
        output_dir: Directory to save enhanced images
        log_file: Path to JSON diagnostics log
        gallery_path: Path to save benchmark gallery
        verbose: Print detailed progress messages
    
    Returns:
        Dictionary with pipeline results
    """
    # Ensure directories exist
    ensure_directories()
    
    # Get list of images to process
    image_files = get_image_files(input_dir)
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return {
            "total_images": 0,
            "processed": 0,
            "failed": 0,
            "gallery_generated": False
        }
    
    print("=" * 80)
    print("IMAGE ENHANCEMENT PIPELINE")
    print("=" * 80)
    print(f"Processing {len(image_files)} images from: {input_dir}")
    print()
    
    # Process each image
    processed = 0
    failed = 0
    all_metrics = []
    
    for i, image_file in enumerate(image_files, 1):
        output_file = Path(output_dir) / image_file.name
        
        metrics, treatments, success = process_image(
            str(image_file),
            str(output_file),
            verbose=verbose
        )
        
        if success:
            processed += 1
            
            # Create log entry
            _, report = diagnose_and_prescribe(metrics)
            log_entry = create_log_entry(image_file.name, metrics, report)
            all_metrics.append(log_entry)
            
            # Log to JSON
            log_diagnostics(log_entry, log_file)
        else:
            failed += 1
    
    print()
    print("=" * 80)
    print(f"RESULTS: {processed}/{len(image_files)} processed, {failed} failed")
    print("=" * 80)
    
    # Generate gallery
    print()
    print("Generating benchmark gallery...")
    gallery_success = build_gallery_advanced(
        input_dir, output_dir, gallery_path,
        metrics_list=all_metrics,
        title="Image Enhancement Benchmark Gallery"
    )
    
    if gallery_success:
        print(f"✓ Gallery saved: {gallery_path}")
    else:
        print(f"✗ Gallery generation failed")
    
    print()
    print("Pipeline complete!")
    print(f"  • Enhanced images: {output_dir}")
    print(f"  • Diagnostics log: {log_file}")
    print(f"  • Benchmark gallery: {gallery_path}")
    
    return {
        "total_images": len(image_files),
        "processed": processed,
        "failed": failed,
        "gallery_generated": gallery_success,
        "output_dir": output_dir,
        "log_file": log_file,
        "gallery_path": gallery_path
    }


if __name__ == "__main__":
    # Run the full pipeline
    results = main()
    
    # Exit with appropriate code
    exit(0 if results["processed"] > 0 else 1)
