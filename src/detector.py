"""Logika inference YOLO untuk mendeteksi kursi."""

from pathlib import Path

import cv2
from ultralytics import YOLO

from config import CLASS_NAMES, CONFIDENCE_THRESHOLD, IMAGE_SIZE, IOU_THRESHOLD, MODEL_PATH


class ChairDetector:
    """Memuat model sekali dan menjalankan inference untuk setiap frame webcam."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model tidak ditemukan: {model_path}\n"
                "Salin file best.pt hasil training ke folder models."
            )
        self.model = YOLO(model_path)

    def detect(self, frame):
        """Mendeteksi kursi pada frame BGR dan mengembalikan frame + jumlah class."""
        result = self.model(
            frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD,
            imgsz=IMAGE_SIZE, verbose=False,
        )[0]

        annotated_frame = frame.copy()
        counts = {"occupied": 0, "unoccupied": 0}

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            if class_id not in CLASS_NAMES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_name = CLASS_NAMES[class_id]
            color = (0, 200, 0) if class_id == 0 else (0, 165, 255)

            if class_id == 0:
                counts["occupied"] += 1
            else:
                counts["unoccupied"] += 1

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        counts["total"] = counts["occupied"] + counts["unoccupied"]
        return annotated_frame, counts
