import cv2
import os
import sys
# 경로 수정
sys.path.append(os.path.abspath("C:\Users\Jong Min Lee\OneDrive\Desktop\github\refrigerator"))

import pytesseract
import numpy as np
from PIL import ImageGrab

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to capture the screen
def capture_screen(bbox=(300, 300, 1500, 1000)):
    cap_scr = np.array(ImageGrab.grab(bbox))
    cap_scr = cv2.cvtColor(cap_scr, cv2.COLOR_RGB2BGR)
    return cap_scr

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Capture the screen as an alternative source (uncomment this if needed)
    # frame = capture_screen()

    # Perform text detection on the frame
    recognized_text = pytesseract.image_to_string(frame)

    # Display the recognized text on the frame in green color
    cv2.putText(frame, recognized_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Real-Time Text Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
