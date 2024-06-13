
import cv2
from mmocr.apis import MMOCRInferencer
import time
import re
def preprocess_image(image):
  # 흑백 변환 유지 (필요)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(image)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    thresh = cv2.adaptiveThreshold(opened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

# 유통기한 추출 함수
def extract_expiration_date(text):
    pattern = r'\b\d{8}\b'  # yyyymmdd 형태
    matches = re.findall(pattern, text)
    return matches
infer = MMOCRInferencer(det='dbnetpp', rec='svtr-small')


cap = cv2.VideoCapture(0)  # 웹캠 0 사용 (변경 가능)
capture_interval = 5
start_time = time.time()

while True:
    current_time = time.time()
    ret, frame = cap.read()

    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 원하는 영역 정의 (예시: 중앙 100x100 영역)
    x = int(frame.shape[1] / 2) - 50
    y = int(frame.shape[0] / 2) - 50
    w = 450
    h = 450
    cropped_img = gray[y:y+h, x:x+w]
    preprocessed_frame = preprocess_image(cropped_img)
    
    img_path = 'current_frame.png'
    cv2.imwrite(img_path, preprocessed_frame)
    result = infer(img_path)
    words_list = result['predictions'][0]['rec_texts']
    recognized_words = [re.sub(r'<[^>]*>', '', word) for word in words_list]
    expiration_dates = extract_expiration_date(' '.join(recognized_words))
    print("Recognized Words:", recognized_words)
    print("Extracted Expiration Dates:", expiration_dates)
    if expiration_dates:
        print("Expiration Dates:", expiration_dates)
        # 화면 크기 조정
    

    cv2.imshow('Webcam', preprocessed_frame)
    cv2.resizeWindow('Webcam', cropped_img.shape[1], cropped_img.shape[0])
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()