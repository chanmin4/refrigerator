import cv2
import pytesseract
import sys

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Ensure UTF-8 output in the console
if sys.version_info[0] >= 3:
    sys.stdout.reconfigure(encoding='utf-8')

# Initialize the webcam
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set language configuration for Tesseract (Korean)
config = r'--oem 3 --psm 6 -l kor'

while True:
    ret, frame = capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Perform text recognition on the binary image
    data = pytesseract.image_to_data(binary, config=config, output_type=pytesseract.Output.DICT)
    
    # Initialize an empty string to collect texts
    detected_text = ""
    
    # Filter out low confidence text detections
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:  # Adjust confidence threshold
            text = data['text'][i].encode('utf-8').decode('utf-8', 'ignore')  # Handle encoding
            detected_text += text + " "  # Append text with a space

    # Print all detected text on a single line
    print(detected_text.strip())

    # Display the resulting frame
    cv2.imshow('Video Frame with Text', frame)
    
    # Press 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
