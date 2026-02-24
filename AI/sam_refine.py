import streamlit as st
import cv2
import numpy as np
import tempfile
from segment_anything import sam_model_registry, SamPredictor
from object_d import detect_clothes

st.set_page_config(page_title="YOLO + SAM Video Segmentation", page_icon="👕")
st.title("👕 DeepFashion YOLO + SAM (Stable Color Video Mode)")

# ======================
# LOAD SAM
# ======================
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

uploaded_file = st.file_uploader("Upload Video", type=["mp4","avi","mov","mkv"])

if uploaded_file and st.button("🔍 Process Video"):

    # Save video
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    input_path = tfile.name
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_count = 0

    # Store smoothed colors per label
    color_memory = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        overlay = frame.copy()

        # ======================
        # YOLO DETECTION
        # ======================
        detections = detect_clothes(frame)

        # ======================
        # SAM REFINEMENT
        # ======================
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        for det in detections:

            x1, y1, x2, y2 = det["box"]
            label = det["label"]

            input_box = np.array([x1, y1, x2, y2])

            masks, _, _ = predictor.predict(
                box=input_box,
                multimask_output=False
            )

            mask = masks[0]

            # ======================
            # REAL CLOTH COLOR EXTRACTION
            # ======================
            cloth_pixels = frame[mask]

            if len(cloth_pixels) > 0:
                avg_color = np.mean(cloth_pixels, axis=0)
                new_color = avg_color.astype(int)
            else:
                new_color = np.array([128,128,128])

            # ======================
            # TEMPORAL COLOR SMOOTHING
            # ======================
            if label in color_memory:
                prev_color = color_memory[label]
                smooth_color = (0.7 * prev_color + 0.3 * new_color).astype(int)
            else:
                smooth_color = new_color

            color_memory[label] = smooth_color
            color = smooth_color.tolist()

            # ======================
            # APPLY MASK
            # ======================
            colored_mask = np.zeros_like(frame)
            colored_mask[mask] = color

            overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.6, 0)

            cv2.putText(
                overlay,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        out.write(overlay)

        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
        status_text.text(f"Processing {frame_count}/{total_frames}")

    cap.release()
    out.release()

    progress_bar.empty()
    status_text.empty()

    st.success("✅ Video Processing Complete!")

    with open(output_path, "rb") as f:
        video_data = f.read()

    st.download_button(
        "📥 Download SAM Refined Video",
        video_data,
        file_name="sam_refined_output.mp4",
        mime="video/mp4",
        use_container_width=True
    )