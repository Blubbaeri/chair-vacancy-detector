"""Menjalankan Chair Vacancy Detector secara realtime melalui webcam."""

import sys
import time

import cv2

from config import WEBCAM_INDEX
from detector import ChairDetector


def draw_information(frame, counts, fps: float) -> None:
    """Menampilkan ringkasan detection di sisi kiri atas frame."""
    lines = [
        f"Occupied: {counts['occupied']}",
        f"Unoccupied: {counts['unoccupied']}",
        f"Total: {counts['total']}",
        f"FPS: {fps:.1f}",
    ]
    for index, text in enumerate(lines):
        cv2.putText(frame, text, (15, 30 + index * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    try:
        detector = ChairDetector()
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        sys.exit(1)
    except Exception as error:
        print(f"ERROR: Model gagal dimuat: {error}")
        sys.exit(1)

    camera = cv2.VideoCapture(WEBCAM_INDEX)
    if not camera.isOpened():
        print(
            f"ERROR: Webcam dengan index {WEBCAM_INDEX} tidak dapat dibuka.\n"
            "Pastikan webcam tersambung, tidak sedang digunakan aplikasi lain, "
            "atau ubah WEBCAM_INDEX di src/config.py."
        )
        camera.release()
        sys.exit(1)

    print("Webcam aktif. Tekan Q atau ESC untuk keluar.")
    previous_time = time.perf_counter()
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("ERROR: Frame dari webcam tidak dapat dibaca.")
                break

            output_frame, counts = detector.detect(frame)
            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            fps = 1 / elapsed if elapsed > 0 else 0.0
            previous_time = current_time

            draw_information(output_frame, counts, fps)
            cv2.imshow("Chair Vacancy Detector", output_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # Q atau ESC
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
