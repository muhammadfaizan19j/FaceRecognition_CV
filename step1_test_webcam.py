# We import cv2 — this is the name Python uses for OpenCV
# Think of 'import' as opening a toolbox before using its tools
import cv2

# VideoCapture(0) tells OpenCV: "open camera number 0"
# 0 means your built-in/default webcam
# If you have two cameras, try VideoCapture(1) for the second one
camera = cv2.VideoCapture(0)

# Check if camera opened successfully
# .isOpened() returns True if camera is ready, False if there was a problem
if not camera.isOpened():
    print("ERROR: Cannot open camera. Check if it is connected.")
    exit()  # Stop the program if camera failed

print("Camera is open! Look at the window. Press Q to quit.")

# 'while True' creates an infinite loop — it runs forever until we 'break' out
# We need this because video = continuous stream of frames
while True:

    # camera.read() grabs the next frame from the webcam
    # It returns TWO things:
    #   success = True if the frame was captured OK, False if error
    #   frame   = the actual image (a big grid of pixel colors)
    success, frame = camera.read()

    # If reading failed (camera disconnected, etc.), stop
    if not success:
        print("Failed to read from camera.")
        break

    # imshow() means "image show" — displays the frame in a pop-up window
    # First argument: the window title (any text you like)
    # Second argument: the frame (image) to display
    cv2.imshow("My Webcam - Press Q to quit", frame)

    # waitKey(1) pauses for 1 millisecond and checks if a key was pressed
    # '& 0xFF' is a technical trick to make it work on all computers
    # ord('q') converts the letter 'q' to its numeric code (113)
    # So: if the Q key was pressed, break out of the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Always release the camera when done — this frees it for other programs
camera.release()

# Close all OpenCV windows that were opened
cv2.destroyAllWindows()
print("Camera closed. Program finished.")