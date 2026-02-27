import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from segment_anything import sam_model_registry, SamPredictor
import torch

st.set_page_config(page_title="Cloth Transfer", page_icon="👕")
st.title("Stable Clothing Transfer - High Accuracy")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    model = YOLO("deepfashion2_yolov8s-seg.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    sam.to(device)

    predictor = SamPredictor(sam)
    return model, predictor

model, predictor = load_models()

video_file = st.file_uploader("Upload Video", type=["mp4"])
source_file = st.file_uploader("Upload Source Image", type=["jpg","png"])

source_cloth = None
valid_source = False

# ---------------- EXTRACT SOURCE CLOTH ----------------
if source_file:
    file_bytes = np.asarray(bytearray(source_file.read()), dtype=np.uint8)
    source_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results_src = model(source_img, conf=0.6)

    if results_src[0].masks is None:
        st.error("❌ No clothing detected.")
    else:
        masks = results_src[0].masks.data.cpu().numpy()
        boxes = results_src[0].boxes

        for mask, box in zip(masks, boxes):
            cls = int(box.cls[0])
            if cls not in [0,1]:
                continue

            mask = (mask > 0.5).astype(np.uint8)
            mask = cv2.resize(mask,
                (source_img.shape[1], source_img.shape[0]),
                interpolation=cv2.INTER_NEAREST)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cloth = source_img.copy()
            cloth[mask == 0] = 0
            source_cloth = cloth[y1:y2, x1:x2]

            if source_cloth.size == 0:
                continue

            # Inpaint small black holes
            inpaint_mask = (source_cloth.sum(axis=-1) == 0).astype(np.uint8) * 255
            source_cloth = cv2.inpaint(
                source_cloth,
                inpaint_mask,
                3,
                cv2.INPAINT_TELEA
            )

            valid_source = True
            st.success("✅ Shirt extracted successfully!")
            break

# ---------------- PROCESS VIDEO ----------------
if video_file and st.button("Process"):
    if not valid_source:
        st.error("❌ Upload valid shirt image first.")
    else:
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
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output = frame.copy()
            results = model(frame, conf=0.5)

            if results[0].boxes is not None:

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                predictor.set_image(frame_rgb)

                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls not in [0,1]:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    input_box = np.array([x1, y1, x2, y2])

                    masks_sam, _, _ = predictor.predict(
                        box=input_box,
                        multimask_output=False
                    )

                    mask = (masks_sam[0] > 0.5).astype(np.uint8) * 255

                    # ---------------- EDGE SHARPENING ----------------
                    kernel = np.ones((5,5), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                    # Slight feather only on border (not full blur)
                    mask = cv2.GaussianBlur(mask, (7,7), 2)

                    # Resize cloth
                    cloth_resized = cv2.resize(source_cloth,
                                               (x2 - x1, y2 - y1))

                    # Create full frame mask
                    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    full_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

                    # Create cloth canvas
                    cloth_canvas = frame.copy()
                    cloth_canvas[y1:y2, x1:x2] = cloth_resized

                    center = ((x1 + x2)//2, (y1 + y2)//2)

                    # ---------------- POISSON BLENDING ----------------
                    output = cv2.seamlessClone(
                        cloth_canvas,
                        frame,
                        full_mask,
                        center,
                        cv2.NORMAL_CLONE
                    )

            out.write(output)

            frame_count += 1
            progress.progress(frame_count / total_frames)

        cap.release()
        out.release()

        st.success("✅ Processing Complete!")

        with open(out_path, "rb") as f:
            st.download_button("Download Result", f.read(), file_name="output.mp4")