import cv2
import os
import pandas as pd
from deepface import DeepFace
from datetime import datetime
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
# LOAD YOLO MODEL
# ──────────────────────────────────────────────────────────────

yolo = YOLO("yolov8n.pt")

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────

DB_PATH = "student_faces"
MODEL = "Facenet"
METRIC = "cosine"
THRESHOLD = 0.4
RECOGNIZE_EVERY = 90

# ──────────────────────────────────────────────────────────────
# ATTENDANCE SETUP
# ──────────────────────────────────────────────────────────────

os.makedirs("attendance_records", exist_ok=True)

today_str = datetime.now().strftime("%Y-%m-%d")
csv_path = os.path.join("attendance_records", f"{today_str}.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])

already_marked = set(df["Name"].tolist())

def mark_attendance(name):

    global df, already_marked

    if name in already_marked:
        return

    now = datetime.now()

    row = {
        "Name": name,
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "Status": "Present"
    }

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(csv_path, index=False)

    already_marked.add(name)

    print(f"MARKED: {name}")

# ──────────────────────────────────────────────────────────────
# CAMERA
# ──────────────────────────────────────────────────────────────

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Cannot open camera")
    exit()

frame_count = 0

last_name = "Unknown"

print("\nYOLO Attendance System Running...\n")

# ──────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────

while True:

    success, frame = camera.read()

    if not success:
        break

    # Resize for speed
    frame = cv2.resize(frame, (640, 480))

    frame_count += 1

    # ──────────────────────────────────────────────────────────
    # YOLO DETECTION
    # ──────────────────────────────────────────────────────────

    yolo_results = yolo(frame, classes=[0], verbose=False)

    for box in yolo_results[0].boxes:

        confidence = box.conf[0].item()

        if confidence < 0.6:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw orange person box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)

        cv2.putText(
            frame,
            "Person",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 165, 0),
            2
        )

        # ──────────────────────────────────────────────────────
        # RUN DEEPFACE EVERY FEW FRAMES
        # ──────────────────────────────────────────────────────

        if frame_count % RECOGNIZE_EVERY == 0:

            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            temp = "_temp.jpg"

            cv2.imwrite(temp, person_crop)

            try:

                results = DeepFace.find(
                    img_path=temp,
                    db_path=DB_PATH,
                    model_name=MODEL,
                    distance_metric=METRIC,
                    enforce_detection=False,
                    silent=True
                )

                if results and len(results[0]) > 0:

                    best = results[0].iloc[0]

                    distance = best["distance"]

                    if distance < THRESHOLD:

                        parts = os.path.normpath(
                            best["identity"]
                        ).split(os.sep)

                        name = parts[-2]

                        last_name = name

                        mark_attendance(name)

            except Exception as e:
                print(f"Recognition error: {e}")

            if os.path.exists(temp):
                os.remove(temp)

        # Show recognized name
        cv2.putText(
            frame,
            f"Recognized: {last_name}",
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # ──────────────────────────────────────────────────────────
    # STATUS BAR
    # ──────────────────────────────────────────────────────────

    status = f"Marked Present: {len(already_marked)}"

    cv2.putText(
        frame,
        status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.imshow("YOLO Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ──────────────────────────────────────────────────────────────
# CLEANUP
# ──────────────────────────────────────────────────────────────

camera.release()

cv2.destroyAllWindows()

print(f"\nAttendance saved to: {csv_path}")