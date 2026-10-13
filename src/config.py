"""Konfigurasi aplikasi Chair Vacancy Detector."""

from pathlib import Path

# Folder utama project: Chair-Vacancy-Detector/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model hasil training. Salin best.pt ke folder models/.
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

# Pengaturan inference
CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45
WEBCAM_INDEX = 0
IMAGE_SIZE = 640

# Class harus sama dengan class saat training.
CLASS_NAMES = {0: "Occupied-chair", 1: "Unoccupied-chair"}
