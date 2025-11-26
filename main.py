import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

# ---------- CONFIG ----------
MODEL_PATH = r"model/best.pt"
OUT_DIR = r"batch_results"
CONF = 0.45


# ---------- END CONFIG ----------

def process_image_folder(model, input_dir):
    inp = Path(input_dir)
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in inp.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not images:
        print("No images found in", inp)
        return

    print(f"Running inference on {len(images)} images...")
    for p in images:
        results = model.predict(source=str(p), conf=CONF)
        r = results[0]
        labels = [r.names[int(c)] for c in r.boxes.cls] if len(r.boxes) else []
        print(f"{p.name} -> {len(labels)} objects, classes: {labels}")
        process_inputs(r)

    print("Saved annotated images to:", out.resolve())


def send_ambulance_signal():
    import serial
    import time

    ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=1)
    time.sleep(2)  # wait for Arduino reset
    ser.write(b"1\n")
    print(f"sent 1 serially")
    time.sleep(2)


def process_inputs(r):
    result = []
    for box in r.boxes:
        cls_id = int(box.cls)
        result.append({
            "class_id": cls_id,
            "class_name": r.names[cls_id],
            "confidence": float(box.conf)
        })

    for ele in result:
        if ele["confidence"] < 0.7:
            continue
        print(ele)
        if ele["class_name"] == "emergency":
            if is_serial:
                send_ambulance_signal()
                break


def process_camera(model, cam_id):
    print(f"Opening webcam {cam_id}...")
    cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        print("❌ Could not open webcam:", cam_id)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        results = model.predict(frame, conf=CONF)
        r = results[0]
        annotated = r.plot()
        cv2.imshow("Webcam Detection", annotated)
        process_inputs(r)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    global is_serial

    # ---------------------------
    # ARGUMENT PARSER
    # ---------------------------
    parser = argparse.ArgumentParser(
        description="Run YOLO on folder, webcam, or video"
    )

    parser.add_argument(
        "--folder",
        type=str,
        help="Path to folder of images"
    )

    parser.add_argument(
        "--webcam",
        type=str,
        help="Webcam ID (e.g., 0)"
    )

    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file"
    )

    parser.add_argument(
        "--serial",
        action="store_true",
        help="Enable serial mode"
    )

    args = parser.parse_args()

    # ---------------------------
    # LOAD MODEL
    # ---------------------------
    model = YOLO(MODEL_PATH)

    # STORE FLAG
    is_serial = args.serial

    # ---------------------------
    # ARGUMENT HANDLING
    # ---------------------------
    if args.folder:
        process_image_folder(model, args.folder)
        return

    if args.webcam:
        process_camera(model, args.webcam)
        return

    # If no valid mode:
    parser.print_help()


if __name__ == "__main__":
    is_serial = False
    main()
