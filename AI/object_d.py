# object_d.py

from ultralytics import YOLO

model = YOLO("deepfashion2_yolov8s-seg.pt")

def detect_clothes(frame):

    results = model.predict(
        frame,
        conf=0.4,
        imgsz=960,
        verbose=False
    )

    detections = []

    if results[0].boxes is not None:
        for box in results[0].boxes:

            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "label": label,
                "confidence": conf,
                "box": [x1, y1, x2, y2]
            })

    return detections