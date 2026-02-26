import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import mediapipe as mp

st.set_page_config(page_title="Step 2 - Pose Detection", page_icon="🧠")
st.title("🧠 Step 2: Label-wise + Combined Pose Detection")

# =========================
# LOAD MEDIAPIPE POSE
# =========================
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=True,
    smooth_segmentation=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# FILE UPLOADS
# =========================
person_video_file = st.file_uploader("Upload Original Person Video", type=["mp4","avi","mov"])
mask_video_file = st.file_uploader("Upload Combined OR Label Mask Video", type=["mp4","avi","mov"])

if person_video_file and mask_video_file and st.button("🚀 Start Pose Detection"):

    # Save videos temporarily
    person_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    person_temp.write(person_video_file.read())
    person_path = person_temp.name

    mask_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    mask_temp.write(mask_video_file.read())
    mask_path = mask_temp.name

    cap = cv2.VideoCapture(person_path)
    mask_cap = cv2.VideoCapture(mask_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_folder = "pose_outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Combined pose writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    combined_writer = cv2.VideoWriter(
        os.path.join(output_folder, "combined_pose.mp4"),
        fourcc,
        fps,
        (width, height)
    )

    # Label-wise writer dictionary
    label_writers = {}

    progress = st.progress(0)
    frame_count = 0

    prev_landmarks = None

    while True:
        ret, frame = cap.read()
        mask_ret, mask_frame = mask_cap.read()

        if not ret or not mask_ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        combined_frame = frame.copy()

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            current_points = []
            for lm in landmarks:
                x = int(lm.x * width)
                y = int(lm.y * height)
                current_points.append((x, y))

            # Manual smoothing
            if prev_landmarks is not None:
                smoothed_points = []
                for i in range(len(current_points)):
                    px, py = prev_landmarks[i]
                    cx, cy = current_points[i]
                    smooth_x = int(0.7 * px + 0.3 * cx)
                    smooth_y = int(0.7 * py + 0.3 * cy)
                    smoothed_points.append((smooth_x, smooth_y))
            else:
                smoothed_points = current_points

            prev_landmarks = smoothed_points

            # Draw on combined frame
            for connection in mp_pose.POSE_CONNECTIONS:
                x1, y1 = smoothed_points[connection[0]]
                x2, y2 = smoothed_points[connection[1]]
                cv2.line(combined_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for x, y in smoothed_points:
                cv2.circle(combined_frame, (x, y), 3, (0, 0, 255), -1)

        # Save combined pose
        combined_writer.write(combined_frame)

        # =============================
        # Label-wise Pose (Mask-based)
        # =============================
        gray_mask = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)

        masked_pose_frame = cv2.bitwise_and(combined_frame, combined_frame, mask=binary_mask)

        label_name = "label_pose"

        if label_name not in label_writers:
            label_writers[label_name] = cv2.VideoWriter(
                os.path.join(output_folder, f"{label_name}.mp4"),
                fourcc,
                fps,
                (width, height)
            )

        label_writers[label_name].write(masked_pose_frame)

        frame_count += 1
        progress.progress(frame_count / total_frames)

    cap.release()
    mask_cap.release()
    combined_writer.release()

    for writer in label_writers.values():
        writer.release()

    st.success("✅ Label-wise + Combined Pose Videos Saved!")
    st.success(f"📁 Saved in folder: {output_folder}")

    # Show outputs
    for file in os.listdir(output_folder):
        if file.endswith(".mp4"):
            st.subheader(file)
            st.video(os.path.join(output_folder, file))