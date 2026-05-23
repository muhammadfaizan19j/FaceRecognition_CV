import cv2   # camera and image tools
import os    # for working with folders and file paths

# ── Ask for student name ────────────────────────────────────────
# input() pauses the program and waits for you to type something
# Use underscores instead of spaces: "Ali_Khan" not "Ali Khan"
student_name = input("Enter student name (e.g. Ali_Khan): ").strip()

# Build the path for this student's folder
# os.path.join builds the correct path for Windows: "student_faces\Ali_Khan"
student_folder = os.path.join("student_faces", student_name)

# Create the folder if it does not exist yet
# exist_ok=True means: do not crash if folder already exists
os.makedirs(student_folder, exist_ok=True)
print(f"Saving photos to: {student_folder}")

# ── Open camera ────────────────────────────────────────────────
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Cannot open camera!")
    exit()

# We will count how many photos we have saved so far
photo_count = 0
MAX_PHOTOS = 5   # how many photos to take per student

print(f"Ready! Press SPACE to take a photo. Need {MAX_PHOTOS} photos. Press Q to quit early.")

while True:
    success, frame = camera.read()
    if not success:
        break

    # Draw a guide rectangle so student knows where to position their face
    # cv2.rectangle(image, top-left corner, bottom-right corner, color BGR, thickness)
    h, w = frame.shape[:2]   # get height and width of the frame
    cx, cy = w // 2, h // 2  # center of the frame
    cv2.rectangle(frame, (cx-100, cy-120), (cx+100, cy+120), (0, 255, 0), 2)

    # Show instructions on the screen using cv2.putText()
    # Arguments: image, text string, position (x,y), font, scale, color BGR, thickness
    cv2.putText(frame,
                f"Student: {student_name}  Photos: {photo_count}/{MAX_PHOTOS}",
                (10, 30),                    # position: 10px from left, 30px from top
                cv2.FONT_HERSHEY_SIMPLEX,    # a clean readable font
                0.7,                         # font size (1.0 = normal, 0.7 = slightly smaller)
                (0, 255, 0),                 # green color in BGR format
                2)                           # line thickness in pixels

    cv2.putText(frame, "SPACE=Capture  Q=Quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Enroll Student - Align face in box", frame)

    key = cv2.waitKey(1) & 0xFF

    # SPACE key has code 32 — when pressed, save the current frame as an image
    if key == 32:
        photo_count += 1   # increase our counter by 1

        # Build filename: "img_001.jpg", "img_002.jpg", etc.
        # zfill(3) pads the number with zeros: 1 becomes "001"
        filename = f"img_{str(photo_count).zfill(3)}.jpg"
        save_path = os.path.join(student_folder, filename)

        # cv2.imwrite() saves the image to your hard drive
        cv2.imwrite(save_path, frame)
        print(f"Saved photo {photo_count}: {save_path}")

        # If we have enough photos, stop automatically
        if photo_count >= MAX_PHOTOS:
            print(f"Done! {MAX_PHOTOS} photos saved for {student_name}.")
            break

    elif key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()