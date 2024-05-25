import os
import sys
# 경로 수정
sys.path.append(os.path.abspath("C:\\Users\\Jong Min Lee\\OneDrive\\Desktop\\github\\refrigerator\\realsense"))

import numpy as np
import tensorflow as tf
import cv2
import pyrealsense2 as rs
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from PIL import ImageFont

# 폰트 로드
font = ImageFont.truetype("arial.ttf", 24)
print(dir(font))  # 폰트 객체에서 사용 가능한 모든 메소드와 속성을 출력합니다.

# 카메라 스트림 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

print("[INFO] Starting streaming...")
pipeline.start(config)
print("[INFO] Camera ready.")

# 모델 및 라벨 경로
PATH_TO_MODEL_DIR = "./saved_model/"
PATH_TO_LABELS = "./mscoco_label_map.pbtxt"
NUM_CLASSES = 101

# 라벨 맵 로드
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# TensorFlow 모델 로드
detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)

# 객체 감지 함수
# 객체 감지 함수
# 객체 감지 함수
# 객체 감지 함수
def run_inference_for_single_image(model, image):
    # 이미지 크기 조정 및 변환
    resized_image = cv2.resize(image, (224, 224))
    resized_image = np.asarray(resized_image, dtype=np.uint8)

    # float32로 형식 변환 후 텐서로 변환
    input_tensor = tf.convert_to_tensor(resized_image, dtype=tf.float32)

    # 모델 호출
    detections = model(input_tensor[tf.newaxis, ...])

    # num_detections 값 평가 및 추출
    num_detections = 100  # 기본값으로 100 설정
    if isinstance(detections, dict) and 'num_detections' in detections.keys():
        num_detections = int(tf.squeeze(detections['num_detections']).numpy())

    # 감지된 객체 정보 추출
    detection_boxes = detections['detection_boxes'][0, :num_detections].numpy()
    detection_classes = detections['detection_classes'][0, :num_detections].numpy()
    detection_scores = detections['detection_scores'][0, :num_detections].numpy()

    # 결과 사전 생성
    detections = {
        'detection_boxes': detection_boxes,
        'detection_classes': detection_classes,
        'detection_scores': detection_scores
    }

    return detections


# 객체 감지 및 시각화
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    # 이미지를 numpy 배열로 변환
    color_image = np.asanyarray(color_frame.get_data())

    # 객체 감지
    detections = run_inference_for_single_image(detect_fn, color_image)

    # 결과 시각화
    vis_util.visualize_boxes_and_labels_on_image_array(
        color_image,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    # 화면에 표시
    cv2.imshow('RealSense Object Detection', color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 시 카메라 및 윈도우 닫기
pipeline.stop()
cv2.destroyAllWindows()
