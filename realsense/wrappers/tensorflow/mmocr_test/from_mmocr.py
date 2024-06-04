import cv2
from mmocr.apis import MMOCRInferencer
import time
import re

# 미리 저장된 단어
saved_words = ["pepsi", "25", "04", "02", "24", "09", "18"]

# MMOCR 인스턴스 초기화
infer = MMOCRInferencer(det='dbnetpp', rec='svtr-small')

# 웹캠 캡처를 위한 VideoCapture 객체 생성
cap = cv2.VideoCapture(0)

# 캡처 주기를 설정 (초)
capture_interval = 5

# 프로그램 시작 시간 기록
start_time = time.time()

while True:
    # 현재 시간 기록
    current_time = time.time()

    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()

    # 프레임 읽기에 실패한 경우 루프를 종료
    if not ret:
        break

    # 현재 시간이 시작 시간에서 캡처 간격의 배수일 때마다 캡처하고 텍스트 인식 수행
    if current_time - start_time >= capture_interval:
        # 현재 시간으로 파일명 설정
        img_path = f'captured_image_{int(current_time)}.png'

        # 프레임 저장
        cv2.imwrite(img_path, frame)
        print(f"Image saved as {img_path}")

        # 텍스트 인식 수행
        result = infer(img_path)
        words_list = result['predictions'][0]['rec_texts']

        # 매칭된 단어 찾기 (특수 문자 제거 후)
        recognized_words = [re.sub(r'<[^>]*>', '', word) for word in words_list]
        matched_words = [word for word in recognized_words if word in saved_words]

        # 인식된 모든 단어 출력
        print("Recognized Words:", recognized_words)

        # 매칭된 단어 출력
        if matched_words:
            print("Matched Words:", matched_words)

        # 프로그램 시작 시간 업데이트
        start_time = time.time()

    # 프레임을 윈도우에 표시
    cv2.imshow('Webcam', frame)

    # 'q' 키를 누르면 웹캠 영상을 종료하고 루프를 종료
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# 웹캠 해제
cap.release()
cv2.destroyAllWindows()
