import cv2
import pytesseract
from PIL import Image

# Tesseract 실행 파일 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 웹캠 초기화
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # 너비 설정
cap.set(4, 480)  # 높이 설정

while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        print("웹캠에서 이미지를 가져오는 데 실패했습니다.")
        break

    # OpenCV 프레임을 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # RGB 프레임을 PIL 이미지로 변환
    pil_img = Image.fromarray(rgb_frame)

    # PIL 이미지에서 텍스트 인식 수행
    recognized_text = pytesseract.image_to_string(pil_img)

    # 프레임에 인식된 텍스트 표시
    cv2.putText(frame, recognized_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 콘솔에 인식된 텍스트 출력
    print(recognized_text)

    # 결과 표시
    cv2.imshow("Real-Time Text Detection", frame)

    # 'q'를 누르면 루프에서 벗어남
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 OpenCV 창 닫기
cap.release()
cv2.destroyAllWindows()
