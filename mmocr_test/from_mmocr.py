from mmocr.apis import MMOCRInferencer
import cv2
import time

# OCR 인퍼런서 초기화
infer = MMOCRInferencer(det='dbnetpp', rec='svtr-small')

def continuous_capture():
    while True:
        # 웹캠에서 이미지 캡처
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            # 메모리상에서 이미지 사용
            cv2.imwrite('temp_image.png', frame)  # 임시 파일로 저장

            # OCR로 텍스트 추출
            result = infer('temp_image.png', return_vis=True)
            words_list = result['predictions'][0]['rec_texts']
            print(words_list)

        cap.release()
        time.sleep(5)  # 5초 대기

if __name__ == "__main__":
    continuous_capture()
