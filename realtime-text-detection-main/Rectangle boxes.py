import cv2
import pytesseract

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to capture the screen
def capture_screen(bbox=(300, 300, 1500, 1000)):
    cap_scr = cv2.cvtColor(cv2.VideoCapture(0).read()[1], cv2.COLOR_RGB2BGR)
    return cap_scr

while True:
    # Read a frame from the webcam
    frame = capture_screen()

    # Perform bounding box detection using Tesseract's built-in capabilities
    d = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 0:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Real-Time Text Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close OpenCV window
cv2.destroyAllWindows()
