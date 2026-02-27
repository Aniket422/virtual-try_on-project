import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from segment_anything import sam_model_registry, SamPredictor
import torch

st.set_page_config(page_title="Cloth Transfer", page_icon="👕")
st.title("Stable Clothing Transfer")

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

# ---------------- INPUT ----------------
video_file = st.file_uploader("Upload Video", type=["mp4"])
source_file = st.file_uploader("Upload Source Image", type=["jpg","png"])

# ---------------- EXTRACT SOURCE CLOTH ----------------
source_cloth = None
valid_source = False

if source_file:
    file_bytes = np.asarray(bytearray(source_file.read()), dtype=np.uint8)
    source_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results_src = model(source_img, conf=0.6)

    if results_src[0].masks is None:
        st.error("❌ No clothing detected in image. Try a clearer front-facing shirt image.")
    else:
        masks = results_src[0].masks.data.cpu().numpy()
        boxes = results_src[0].boxes

        found = False

        for mask, box in zip(masks, boxes):
            cls = int(box.cls[0])

            # only shirt (classes 0 and 1 in DeepFashion2)
            if cls not in [0,1]:
                continue

            if mask is None or mask.size == 0:
                continue

            mask = (mask > 0.5).astype(np.uint8)

            try:
                mask = cv2.resize(mask,
                    (source_img.shape[1], source_img.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
            except:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cloth = source_img.copy()
            cloth[mask == 0] = 0   # black out background

            # crop shirt
            source_cloth = cloth[y1:y2, x1:x2]

            if source_cloth.size == 0:
                continue

            # ----- INPAINT the source cloth to fill black areas -----
            # Create mask of black pixels (background)
            inpaint_mask = (source_cloth.sum(axis=-1) == 0).astype(np.uint8) * 255
            # Inpaint with Telea algorithm
            source_cloth = cv2.inpaint(source_cloth, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            found = True
            break

        if not found:
            st.error("❌ Could not extract shirt. Try another image.")
        else:
            valid_source = True
            st.success("✅ Shirt extracted and inpainted successfully!")

# ---------------- PROCESS VIDEO ----------------
if video_file and st.button("Process"):
    if not valid_source:
        st.error("❌ Please upload a valid shirt image first.")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

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

                    # Get binary mask from SAM
                    mask = (masks_sam[0] > 0.5).astype(np.float32)

                    # ----- FEATHER the mask for smooth blending -----
                    mask = cv2.GaussianBlur(mask, (21, 21), 11)
                    # Expand to 3 channels
                    mask_3c = np.stack([mask, mask, mask], axis=-1)

                    # Crop mask to bounding box
                    mask_roi = mask[y1:y2, x1:x2]
                    mask_roi_3c = mask_3c[y1:y2, x1:x2]

                    # Resize source cloth to target bounding box size
                    cloth_resized = cv2.resize(source_cloth, (x2 - x1, y2 - y1))

                    # Blend using the feathered mask
                    roi = output[y1:y2, x1:x2]
                    blended = roi * (1 - mask_roi_3c) + cloth_resized * mask_roi_3c
                    output[y1:y2, x1:x2] = blended.astype(np.uint8)

            out.write(output)

            frame_count += 1
            progress.progress(frame_count / total_frames)

        cap.release()
        out.release()

        st.success("✅ Done!")

        with open(out_path, "rb") as f:
            st.download_button("Download Result", f.read(), file_name="output.mp4")
