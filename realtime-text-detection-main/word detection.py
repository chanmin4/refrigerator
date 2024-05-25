import cv2
import pytesseract

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize the webcam
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set language configuration for Tesseract (Korean)
config = r'--oem 3 --psm 6 -l kor'

while True:
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)
    
    # Press 'c' key to capture frame and perform text recognition
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply binary thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Perform text recognition on the binary image
        text = pytesseract.image_to_string(gray, config=config)
        
        # Print recognized text
        print('--------------------------')
        print(text)
        print('--------------------------')
        
        # Display the recognized text on the captured frame
        cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Captured Frame", frame)

    # Press 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
