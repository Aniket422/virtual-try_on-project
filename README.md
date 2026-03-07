# virtual-try_on-project


import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from segment_anything import sam_model_registry, SamPredictor
import torch
import os
import mediapipe as mp
from collections import deque, Counter

st.set_page_config(page_title="Cloth Transfer - Body Rotation", page_icon="👕")
st.title("Clothing Transfer - Body Rotation Detection")

# -------------------------------
# Load models
# -------------------------------
@st.cache_resource
def load_models():
    yolo_model = YOLO("deepfashion2_yolov8s-seg.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    sam.to(device)
    predictor = SamPredictor(sam)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    return yolo_model, predictor, pose

yolo_model, predictor, pose = load_models()

# -------------------------------
# Load shirt strip
# -------------------------------
STRIP_PATH = "product1.jpg"
if not os.path.exists(STRIP_PATH):
    st.error("Strip image not found.")
    st.stop()

strip = cv2.imread(STRIP_PATH)
gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area = 5000
shirt_contours = [c for c in contours if cv2.contourArea(c) > min_area]
shirt_contours = sorted(shirt_contours, key=lambda c: cv2.boundingRect(c)[0])

shirt_images = []
for cnt in shirt_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    crop = strip[y:y+h, x:x+w]
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_crop, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    shirt_images.append((crop, mask))

def display_shirt_views(shirt_images):
    """
    Display the extracted shirt images and their masks in a row.
    
    Args:
        shirt_images (list): List of tuples (shirt_crop, shirt_mask) as returned
                             by the shirt extraction logic.
    """
    st.subheader("Extracted Shirt Views")
    if not shirt_images:
        st.warning("No shirt views found.")
        return

    cols = st.columns(len(shirt_images))
    for i, (img, mask) in enumerate(shirt_images):
        with cols[i]:
            # Convert BGR to RGB for correct display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption=f"View {i}", use_container_width=True)
    
    st.caption("Ensure the order matches: 0=front, 1=left, 2=back, 3=right. "
               "If not, adjust the `orient_to_idx` dictionary accordingly.") 
    
if len(shirt_images) != 4:
    st.warning(f"Expected 4 shirts but found {len(shirt_images)}")
else:
    # Optional: display them
    if st.sidebar.checkbox("Show extracted shirt views", False):
        display_shirt_views(shirt_images)

# -------------------------------
# Orientation detection function
# -------------------------------
def get_orientation(landmarks, vis_thresh=0.5, occ_thresh=0.1):
    """
    Determine orientation from MediaPipe pose landmarks.
    Returns: 'front', 'back', 'left', 'right', or None if ambiguous.
    """
    # Helper functions
    def all_visible(indices, thresh=vis_thresh):
        return all(landmarks[i].visibility >= thresh for i in indices)

    def all_occluded(indices, thresh=occ_thresh):
        return all(landmarks[i].visibility <= thresh for i in indices)

    # Landmark indices (MediaPipe Pose)
    NOSE = 0
    EYES = list(range(1, 6))      # 1-5: eyes inner/outer
    EARS = [7, 8]                  # left ear, right ear
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_HIP, RIGHT_HIP = 23, 24

    # ---- Front ----
    if (all_visible([NOSE] + EYES + EARS) and
        all_visible([LEFT_SHOULDER, RIGHT_SHOULDER])):
        # Optional: check shoulders are roughly horizontal
        if abs(landmarks[LEFT_SHOULDER].y - landmarks[RIGHT_SHOULDER].y) < 0.1:
            return "front"

    # ---- Back ----
    if (all_occluded([NOSE] + EYES + EARS) and
        all_visible([LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP])):
        # Optional depth check: shoulders further than hips (z larger)
        hip_center_z = (landmarks[LEFT_HIP].z + landmarks[RIGHT_HIP].z) / 2
        shoulder_avg_z = (landmarks[LEFT_SHOULDER].z + landmarks[RIGHT_SHOULDER].z) / 2
        if shoulder_avg_z > hip_center_z + 0.05:
            return "back"
        # If depth unreliable, still return back based on occlusion
        return "back"

    # ---- Left ----
    if (all_visible([LEFT_SHOULDER, LEFT_HIP]) and
        all_occluded([RIGHT_SHOULDER, RIGHT_HIP])):
        if landmarks[LEFT_SHOULDER].x > landmarks[RIGHT_SHOULDER].x:
            return "left"

    # ---- Right ----
    if (all_visible([RIGHT_SHOULDER, RIGHT_HIP]) and
        all_occluded([LEFT_SHOULDER, LEFT_HIP])):
        if landmarks[RIGHT_SHOULDER].x > landmarks[LEFT_SHOULDER].x:
            return "right"

    return None

# -------------------------------
# Sidebar settings
# -------------------------------
st.sidebar.header("Calibration")
# Note: angles are no longer used, but we keep the input for compatibility
angles = [
    st.sidebar.number_input("View 0 angle", value=0),
    st.sidebar.number_input("View 1 angle", value=45),
    st.sidebar.number_input("View 2 angle", value=90),
    st.sidebar.number_input("View 3 angle", value=135),
]

smoothing_frames = st.sidebar.slider("Orientation smoothing", 1, 30, 10)
show_pose = st.sidebar.checkbox("Show pose skeleton", True)
manual_mode = st.sidebar.checkbox("Manual override", False)
manual_idx = st.sidebar.selectbox("Manual shirt view", [0, 1, 2, 3], index=0)

# -------------------------------
# Video input
# -------------------------------
video_file = st.file_uploader("Upload Video", type=["mp4"])

if video_file and st.button("Process"):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = cv2.VideoWriter(out_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps,
                          (width, height))

    progress = st.progress(0)
    orientation_history = deque(maxlen=smoothing_frames)   # stores orientations
    last_idx = None   # last valid shirt index

    frame_count = 0
    preview = st.empty()

    # Mapping from orientation to shirt index (adjust order to match your images)
    orient_to_idx = {"front": 0, "left": 1, "back": 2, "right": 3}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb)

        valid_pose = False
        idx = last_idx if last_idx is not None else 0   # fallback

        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark

            # ---- 1. Determine orientation using rules ----
            orientation = get_orientation(lm)

            # ---- 2. Check confidence of key landmarks (≥ 0.9) ----
            key_indices = [0, 11, 12, 23, 24]   # nose, shoulders, hips
            min_conf = min(lm[i].visibility for i in key_indices)
            high_conf = min_conf >= 0.9

            if orientation is not None and high_conf:
                orientation_history.append(orientation)
                valid_pose = True

            # ---- 3. Smooth orientation over recent frames ----
            if valid_pose and len(orientation_history) > 0:
                # Use most frequent orientation in history
                smoothed_orient = Counter(orientation_history).most_common(1)[0][0]
                idx = orient_to_idx[smoothed_orient]
                last_idx = idx   # remember last valid index

            # ---- Draw skeleton if requested ----
            if show_pose:
                mp.solutions.drawing_utils.draw_landmarks(
                    output,
                    pose_results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )

        # ---- Manual override ----
        if manual_mode:
            idx = manual_idx

        # ---- Apply shirt only if valid pose (or manual) ----
        if valid_pose or manual_mode:
            results = yolo_model(frame, conf=0.5)
            if results[0].boxes is not None:
                predictor.set_image(rgb)
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls not in [0, 1]:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    input_box = np.array([x1, y1, x2, y2])
                    masks, _, _ = predictor.predict(box=input_box, multimask_output=False)
                    mask = (masks[0] > 0.5).astype(np.uint8) * 255
                    kernel = np.ones((5,5), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                    shirt_img, shirt_mask = shirt_images[idx]
                    shirt_resized = cv2.resize(shirt_img, (x2-x1, y2-y1))
                    mask_resized = cv2.resize(shirt_mask, (x2-x1, y2-y1),
                                              interpolation=cv2.INTER_NEAREST)
                    mask_crop = mask[y1:y2, x1:x2]
                    mask_crop = cv2.resize(mask_crop,
                                           (shirt_resized.shape[1], shirt_resized.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
                    mask_final = cv2.bitwise_and(mask_crop, mask_resized)
                    mask_float = mask_final.astype(np.float32) / 255.0
                    mask_float = mask_float[..., np.newaxis]

                    roi = output[y1:y2, x1:x2]
                    blended = (roi * (1 - mask_float) + shirt_resized * mask_float).astype(np.uint8)
                    output[y1:y2, x1:x2] = blended

        # ---- Write frame ----
        out.write(output)
        frame_count += 1
        progress.progress(frame_count / total_frames)

        if frame_count % 10 == 0:
            preview.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
                          caption=f"Frame {frame_count} | orientation {smoothed_orient if valid_pose else 'N/A'} | view {idx}",
                          use_container_width=True)

    cap.release()
    out.release()
    st.success("Processing complete")
    with open(out_path, "rb") as f:
        st.download_button("Download video", f.read(), file_name="output.mp4")
