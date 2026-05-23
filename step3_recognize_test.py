import cv2
import os
from deepface import DeepFace

# Path to the folder containing all student subfolders
DB_PATH = "student_faces"

# The model and distance metric we will use throughout the project
MODEL      = "VGG-Face"    # good balance of speed and accuracy
METRIC     = "cosine"      # cosine distance works well with VGG-Face
THRESHOLD  = 0.4             # if distance < 0.4, we accept it as a match
                              # lower = stricter. Tune this if needed.

# ── Open camera ────────────────────────────────────────────────
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Cannot open camera!")
    exit()

print("Press SPACE to capture and recognize. Press Q to quit.")

while True:
    success, frame = camera.read()
    if not success:
        break

    # Show the live camera feed
    cv2.imshow("Recognition Test - SPACE to identify", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:   # SPACE was pressed

        # Save current frame to a temporary file on disk
        # DeepFace.find() needs a file path (or numpy array with enforce_detection=False)
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)
        print("Identifying face... (may take a few seconds)")

        try:
            # Ask DeepFace to search our student_faces folder for the best match
            results = DeepFace.find(
                img_path         = temp_path,
                db_path          = DB_PATH,
                model_name       = MODEL,
                distance_metric  = METRIC,
                enforce_detection = False,
                silent           = True   # suppress progress messages
            )

            # results is a list — one item per face detected in the image
            # results[0] is a pandas DataFrame with columns: identity, distance, etc.
            if results and len(results[0]) > 0:

                # Get the row with the LOWEST distance (best match)
                best_match = results[0].iloc[0]  # iloc[0] = first row

                # Extract the distance value
                distance = best_match["distance"]

                if distance < THRESHOLD:
                    # Extract student name from the file path
                    # Path looks like: student_faces\Ali_Khan\img_001.jpg
                    # os.path.normpath normalizes slashes, .split splits into parts
                    identity_path = best_match["identity"]
                    parts = os.path.normpath(identity_path).split(os.sep)
                    # parts = ['student_faces', 'Ali_Khan', 'img_001.jpg']
                    # parts[-2] = 'Ali_Khan' (second from last)
                    student_name = parts[-2]

                    print(f"MATCH FOUND: {student_name} (distance: {distance:.4f})")
                else:
                    print(f"No confident match. Closest distance: {distance:.4f} (threshold: {THRESHOLD})")
            else:
                print("No face found in image or no students enrolled yet.")

        except Exception as e:
            # If DeepFace has an error, print it instead of crashing
            print(f"DeepFace error: {e}")

        # Delete the temporary file to keep things clean
        if os.path.exists(temp_path):
            os.remove(temp_path)

    elif key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()