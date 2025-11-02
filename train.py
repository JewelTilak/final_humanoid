from pathlib import Path
import cv2
import numpy as np
import sys

# if hasattr(sys, '_MEIPASS'):
#     BASE_DIR = Path(sys._MEIPASS)
# else:
#     BASE_DIR = Path(__file__).resolve().parent


if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).resolve().parent

dataset_path = BASE_DIR / "data"
assets_path = BASE_DIR / "assets"

dataset_path.mkdir(exist_ok=True)


if len(sys.argv) > 1:
    person_name = sys.argv[1].strip()
    print(f"Capturing faces for {person_name}...")
else:
    print("❌ No name provided. Please run from GUI or provide a name argument.")
    sys.exit(0)   

modelFile = str(assets_path / "res10_300x300_ssd_iter_140000.caffemodel")
configFile = str(assets_path / "deploy.prototxt")

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access webcam. Check camera permissions.")
    sys.exit(0)

face_data = []
count = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1

    # Process every 5th frame for efficiency
    if frame_count % 5 != 0:
        cv2.imshow("Face Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Resize and prepare for detection
    small_frame = cv2.resize(frame, (320, 240))
    h, w = small_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(small_frame, 1.0, (300, 300), (104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)

            # Scale to original frame
            scale_x = frame.shape[1] / w
            scale_y = frame.shape[0] / h
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

            face_section = frame[y1:y2, x1:x2]
            if face_section.size == 0:
                continue

            face_section = cv2.resize(face_section, (100, 100))
            face_data.append(face_section)
            count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Count: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Capture", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or count >= 100:
        break

# ===========================
# Save Captured Faces
# ===========================
if face_data:
    face_data = np.array(face_data)
    save_path = dataset_path / f"{person_name}.npy"
    np.save(save_path, face_data)
    print(f"✅ Saved {face_data.shape} for {person_name} in {save_path}")
else:
    print("⚠️ No faces captured. Try again.")

cap.release()
cv2.destroyAllWindows()

