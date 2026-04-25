"""
Streamlit GUI for CVP Image Enhancement Pipeline
A web-based interface for the image enhancement pipeline.
"""

import streamlit as st
from pathlib import Path
import cv2
import json
from PIL import Image
import os

from main import main as run_pipeline, process_image
from utils import load_image, get_image_files, ensure_directories
from diagnosis import diagnose
from controller import diagnose_and_prescribe

# Page configuration
st.set_page_config(
    page_title="CVP Image Enhancement",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = {}
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = None

initialize_session_state()

# Sidebar
st.sidebar.title("🎨 CVP Pipeline")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Select Mode:",
    ["📊 Dashboard", "🔄 Process Single Image", "📤 Batch Upload & Process", "📈 View Results"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Directories")
st.sidebar.markdown("""
- **Input:** `dataset/input/`
- **Output:** `dataset/output/`
- **Gallery:** `gallery/benchmark_grid.png`
- **Logs:** `logs/diagnostics.json`
""")

# Main content
if mode == "📊 Dashboard":
    st.title("🖼️ CVP Image Enhancement Pipeline")
    
    st.markdown("""
    ### Welcome to the Image Enhancement Pipeline
    
    This pipeline processes images through three phases:
    1. **Diagnosis** - Analyze image metrics and identify issues
    2. **Prescription** - Determine optimal treatments
    3. **Treatment** - Apply enhancements in correct order
    """)
    
    st.markdown("---")
    st.subheader("🚀 Getting Started")
    st.markdown("""
    **Upload and enhance your images:**
    1. Go to **Process Single Image** to enhance one image at a time
    2. Go to **Batch Upload & Process** to process multiple images at once
    3. Use **View Results** to download your enhanced images
    """)
    
    st.markdown("---")
    st.subheader("📊 Statistics")
    
    output_dir = Path("dataset/output")
    log_file = Path("logs/diagnostics.json")
    
    output_count = len(list(output_dir.glob("*.[pP][nN][gG]"))) + len(list(output_dir.glob("*.[jJ][pP][gG]"))) + len(list(output_dir.glob("*.[jJ][pP][eE][gG]")))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Processed Images", output_count)
    with col2:
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
                st.metric("Processing Log Entries", len(logs) if isinstance(logs, list) else 1)
        else:
            st.metric("Processing Log Entries", 0)
    with col3:
        gallery_exists = Path("gallery/benchmark_grid.png").exists()
        st.metric("Gallery Generated", "✓" if gallery_exists else "✗")
    
    st.markdown("---")
    st.markdown("""
    💡 **Tips:**
    - Supported formats: PNG, JPG, JPEG, BMP, GIF
    - Batch upload is recommended for processing multiple images
    - Download your enhanced images anytime from the View Results page
    """)


elif mode == "🔄 Process Single Image":
    st.title("🔄 Process Single Image")
    
    input_dir = Path("dataset/input")
    output_dir = Path("dataset/output")
    
    ensure_directories()
    
    st.subheader("Upload Image to Process")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose image(s) to process",
        type=["jpg", "jpeg", "png", "bmp", "gif"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.markdown("---")
        st.subheader(f"Processing {len(uploaded_files)} image(s)...")
        
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            temp_input_path = Path("dataset/input") / uploaded_file.name
            
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner(f"Processing {uploaded_file.name}..."):
                output_file = output_dir / uploaded_file.name
                metrics, treatments, success = process_image(
                    str(temp_input_path),
                    str(output_file),
                    verbose=False
                )
            
            if success:
                st.success(f"✓ {uploaded_file.name} processed successfully!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📥 Original Image")
                    original = Image.open(temp_input_path)
                    st.image(original, width=300)
                
                with col2:
                    st.subheader("📤 Enhanced Image")
                    enhanced = Image.open(output_file)
                    st.image(enhanced, width=300)
                
                # Display metrics and treatments
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Diagnostics")
                    if metrics:
                        for key, value in metrics.items():
                            st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
                
                with col2:
                    st.subheader("🎯 Applied Treatments")
                    if treatments:
                        for i, treatment in enumerate(treatments, 1):
                            st.write(f"{i}. {treatment.replace('_', ' ').title()}")
                    else:
                        st.info("No treatments applied")
                
                # Get and display prescription report
                _, report = diagnose_and_prescribe(metrics)
                st.markdown("---")
                st.subheader("📋 Prescription Report")
                st.write(report)
                
                # Download button
                with open(output_file, "rb") as f:
                    st.download_button(
                        label=f"⬇️ Download {uploaded_file.name}",
                        data=f.read(),
                        file_name=f"enhanced_{uploaded_file.name}",
                        mime="image/png"
                    )
                
                st.markdown("---")
            else:
                st.error(f"✗ Failed to process {uploaded_file.name}")
    else:
        st.info("👆 Upload one or more images to get started")


elif mode == "📤 Batch Upload & Process":
    st.title("📤 Batch Upload & Process")
    
    st.markdown("""
    Upload multiple images to process through the complete pipeline.
    All enhanced images will be available for download.
    """)
    
    input_dir = Path("dataset/input")
    output_dir = Path("dataset/output")
    
    ensure_directories()
    
    st.subheader("📁 Select Images to Process")
    
    # File uploader for batch
    uploaded_files = st.file_uploader(
        "Choose multiple images to process",
        type=["jpg", "jpeg", "png", "bmp", "gif"],
        accept_multiple_files=True,
        key="batch_uploader"
    )
    
    if uploaded_files and len(uploaded_files) > 0:
        st.markdown("---")
        
        st.subheader(f"📊 Ready to Process {len(uploaded_files)} Image(s)")
        
        col1, col2 = st.columns(2)
        with col1:
            verbose = st.checkbox("Show detailed progress", value=False)
        with col2:
            auto_download = st.checkbox("Auto-prepare downloads", value=True)
        
        if st.button("🚀 Process All Uploads", use_container_width=True):
            with st.spinner(f"Processing {len(uploaded_files)} images..."):
                processed = 0
                failed = 0
                all_metrics = []
                download_files = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    # Save uploaded file
                    temp_input_path = input_dir / uploaded_file.name
                    
                    with open(temp_input_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process
                    output_file = output_dir / uploaded_file.name
                    metrics, treatments, success = process_image(
                        str(temp_input_path),
                        str(output_file),
                        verbose=False
                    )
                    
                    if success:
                        processed += 1
                        _, report = diagnose_and_prescribe(metrics)
                        all_metrics.append({
                            "filename": uploaded_file.name,
                            "metrics": metrics,
                            "treatments": treatments
                        })
                        if auto_download:
                            with open(output_file, "rb") as f:
                                download_files[uploaded_file.name] = f.read()
                    else:
                        failed += 1
                    
                    # Update progress
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {idx + 1}/{len(uploaded_files)} - {processed} succeeded, {failed} failed")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
            
            # Display results
            st.markdown("---")
            st.success(f"✓ Batch processing complete! {processed}/{len(uploaded_files)} processed, {failed} failed")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Uploaded", len(uploaded_files))
            with col2:
                st.metric("Successfully Processed", processed)
            with col3:
                st.metric("Failed", failed)
            
            # Display results for each image
            st.markdown("---")
            st.subheader("📊 Processing Results")
            
            for idx, uploaded_file in enumerate(uploaded_files):
                temp_input_path = input_dir / uploaded_file.name
                output_file = output_dir / uploaded_file.name
                
                if output_file.exists():
                    with st.expander(f"✓ {uploaded_file.name}", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Original")
                            original = Image.open(temp_input_path)
                            st.image(original, width=250)
                        
                        with col2:
                            st.subheader("Enhanced")
                            enhanced = Image.open(output_file)
                            st.image(enhanced, width=250)
                        
                        # Find metrics for this image
                        image_meta = next((m for m in all_metrics if m["filename"] == uploaded_file.name), None)
                        if image_meta:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Metrics")
                                for key, value in image_meta["metrics"].items():
                                    st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
                            
                            with col2:
                                st.subheader("Treatments Applied")
                                for i, treatment in enumerate(image_meta["treatments"], 1):
                                    st.write(f"{i}. {treatment.replace('_', ' ').title()}")
                        
                        # Download button
                        with open(output_file, "rb") as f:
                            st.download_button(
                                label=f"⬇️ Download enhanced_{uploaded_file.name}",
                                data=f.read(),
                                file_name=f"enhanced_{uploaded_file.name}",
                                mime="image/png",
                                key=f"download_{idx}"
                            )
                else:
                    with st.expander(f"✗ {uploaded_file.name} - Processing Failed", expanded=False):
                        st.error(f"Failed to process {uploaded_file.name}")
    else:
        st.info("👆 Upload one or more images to get started")


elif mode == "📈 View Results":
    st.title("📈 View Results")
    
    output_dir = Path("dataset/output")
    
    st.subheader("Processed Images")
    output_files = get_image_files(str(output_dir))
    
    if output_files:
        selected_output = st.selectbox(
            "Select an enhanced image:",
            output_files,
            format_func=lambda x: x.name,
            key="output_selector"
        )
        
        img = Image.open(selected_output)
        st.image(img, width=None, caption=selected_output.name)
        
        # Show file info
        file_size = selected_output.stat().st_size / 1024  # KB
        st.caption(f"File size: {file_size:.1f} KB")
        
        # Download button
        with open(selected_output, "rb") as f:
            st.download_button(
                label=f"⬇️ Download {selected_output.name}",
                data=f.read(),
                file_name=selected_output.name,
                mime="image/png"
            )
    else:
        st.info("No enhanced images found. Process images first.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    CVP Image Enhancement Pipeline • Powered by Streamlit
</div>
""", unsafe_allow_html=True)
