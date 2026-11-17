"""Live webcam classification and YOLO detection."""
from pathlib import Path

import cv2
import torch


USE_YOLO = True
CAMERA_INDEX = 0
LOW_LIGHT_THRESHOLD = 60.0
MODEL_DIR = Path(__file__).resolve().parent.parent


def main():
    if USE_YOLO:
        from detect_yolo import GarbageDetector

        detector = GarbageDetector(str(MODEL_DIR / "garbage_detection_best.pt"), conf=0.025)
        model_ft = None
    else:
        from classify import visualize_model_predictions

        model_ft = torch.load(
            MODEL_DIR / "model_ft.pth",
            weights_only=False,
            map_location=torch.device("cpu"),
        )
        model_ft.eval()
        detector = None

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow("Garbage Detection", cv2.WINDOW_NORMAL)
    print("Webcam detection active. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read a frame from the webcam.")
                break

            brightness = cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0]

            if USE_YOLO:
                primary, confidence = detector.get_primary(frame)
                detection_text = (
                    f"{primary} ({confidence:.2f})" if primary else "No detection"
                )
            else:
                detection_text = visualize_model_predictions(model_ft, frame)

            cv2.putText(
                frame,
                detection_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            light_text = f"Brightness: {brightness:.0f}/255"
            light_color = (0, 0, 255) if brightness < LOW_LIGHT_THRESHOLD else (0, 255, 0)
            if brightness < LOW_LIGHT_THRESHOLD:
                light_text += " - low light, add illumination"
            cv2.putText(
                frame,
                light_text,
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                light_color,
                2,
            )
            cv2.imshow("Garbage Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
