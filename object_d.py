import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from collections import Counter

st.set_page_config(page_title="Clothing Segmentor", page_icon="👕")

st.title("👕 Clothing Item Segmentor")
st.write("This model outlines each clothing item with colored masks!")

# Download the segmentation model
@st.cache_resource
def load_model():
    # Direct download URL for the segmentation model
    model_url = "https://huggingface.co/Bingsu/adetailer/resolve/main/deepfashion2_yolov8s-seg.pt"
    model = YOLO(model_url)
    return model

# Clothing categories from DeepFashion2
clothing_categories = {
    0: 'short_sleeved_shirt',
    1: 'long_sleeved_shirt',
    2: 'short_sleeved_outwear',
    3: 'long_sleeved_outwear',
    4: 'vest',
    5: 'sling',
    6: 'shorts',
    7: 'trousers',
    8: 'skirt',
    9: 'short_sleeved_dress',
    10: 'long_sleeved_dress',
    11: 'vest_dress',
    12: 'sling_dress'
}

# Colors for each clothing type
colors = {
    'short_sleeved_shirt': (255, 0, 0),  # Blue
    'long_sleeved_shirt': (0, 0, 255),  # Red
    'short_sleeved_outwear': (255, 165, 0),  # Orange
    'long_sleeved_outwear': (255, 255, 0),  # Yellow
    'vest': (128, 0, 128),  # Purple
    'sling': (255, 192, 203),  # Pink
    'shorts': (0, 255, 0),  # Green
    'trousers': (0, 128, 0),  # Dark Green
    'skirt': (255, 20, 147),  # Deep Pink
    'short_sleeved_dress': (75, 0, 130),  # Indigo
    'long_sleeved_dress': (138, 43, 226),  # Blue Violet
    'vest_dress': (255, 105, 180),  # Hot Pink
    'sling_dress': (255, 182, 193)  # Light Pink
}

# Load model
model = load_model()

uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file and st.button("🔍 Segment Clothing Items", type="primary"):
    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    input_path = tfile.name
    
    # Output path
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    all_detections = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run segmentation
        results = model.track(frame, conf=0.5, iou=0.5, imgsz=768, persist=True)
        
        # Create overlay for masks
        overlay = frame.copy()
        
        # Process results
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Get clothing type
                clothing_type = clothing_categories.get(cls, "unknown")
                
                # Resize mask to frame size
                mask = (mask > 0.6).astype(np.uint8)
                
                # Apply morphological smoothing
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.GaussianBlur(mask.astype(float), (5,5), 0)
                mask = (mask > 0.5).astype(np.uint8)
                
                # Get color for this clothing type
                color = colors.get(clothing_type, (128, 128, 128))
                
                # Create colored mask
                colored_mask = np.zeros_like(frame)
                for c in range(3):
                    colored_mask[:, :, c] = mask * color[c]
                
                # Blend with overlay
                overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.6, 0)
                
                # Add label
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{clothing_type}: {conf:.2f}"
                cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Store detection
                all_detections.append(clothing_type)
        
        # Write frame
        out.write(overlay)
        
        # Update progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
    
    # Clean up
    cap.release()
    out.release()
    
    # Clear progress
    progress_bar.empty()
    status_text.empty()
    
    st.success("✅ Processing complete!")
    
    # Download button
    with open(output_path, 'rb') as f:
        video_data = f.read()
    
    st.download_button(
        label="📥 DOWNLOAD SEGMENTED VIDEO",
        data=video_data,
        file_name="clothing_segmented.mp4",
        mime="video/mp4",
        use_container_width=True
    )
    
    # Show statistics
    if all_detections:
        st.subheader("📊 Clothing Items Detected")
        counts = Counter(all_detections)
        
        # Display in columns
        cols = st.columns(3)
        for i, (item, count) in enumerate(counts.most_common(6)):
            with cols[i % 3]:
                # Get color for this item
                color = colors.get(item, (128,128,128))
                color_hex = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
                st.markdown(f"<span style='color:{color_hex}'>●</span> **{item}**: {count}", unsafe_allow_html=True)

# Info section
st.markdown("---")
st.markdown("""
### 📋 What this model detects:
| Category | Items |
|----------|-------|
| **Shirts** | Short sleeve, Long sleeve |
| **Outerwear** | Short sleeve, Long sleeve, Vest, Sling |
| **Bottoms** | Shorts, Trousers, Skirt |
| **Dresses** | Short sleeve, Long sleeve, Vest dress, Sling dress |

The model creates **colored masks** around each clothing item, not just boxes!
""")