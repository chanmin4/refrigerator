import cv2
import tensorflow as tf
import numpy as np
import uuid
import time
import re
import json
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from wisepaasdatahubedgesdk.EdgeAgent import EdgeAgent
from wisepaasdatahubedgesdk.Model.Edge import EdgeAgentOptions, DCCSOptions, EdgeData, EdgeTag, constant
import sys
sys.path.append('./mmocr_test')
import base64
from mmocr.apis import MMOCRInferencer
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2

# 경로 제거
sys.path.remove('./mmocr_test')

# MMOCR 인스턴스 초기화 및 EdgeAgent 설정
infer = MMOCRInferencer(det='dbnetpp', rec='svtr-small')
edgeAgentOptions = EdgeAgentOptions(nodeId='3607ae3d-5e1e-4171-8706-b6a111fa05ac')
dccsOptions = DCCSOptions(apiUrl='https://api-dccs-ensaas.sa.wise-paas.com/', credentialKey='8d47cc1fab2e0a5207ab7da336ae4atl')
edgeAgentOptions.DCCS = dccsOptions
edgeAgent = EdgeAgent(edgeAgentOptions)
edgeAgent.connect()

def send_class_to_datahub(detected_items):
    edgeData = EdgeData()
    items_json = json.dumps(detected_items)
    tag = EdgeTag(deviceId="volume_camera", tagName="object_recog", value=items_json)
    edgeData.tagList.append(tag)
    edgeAgent.sendData(edgeData)
    print(f"num of items:{len(detected_items)}")
    for item in detected_items:
        print(f"Class Name:{item['class_name']}")
    print(f"Sent detected objects to DataHub: {items_json}")

def send_word_to_datahub(matched_words):
    edgeData = EdgeData()
    items_json = json.dumps(matched_words)
    tag = EdgeTag(deviceId="volume_camera", tagName="words", value=items_json)
    edgeData.tagList.append(tag)
    edgeAgent.sendData(edgeData)
    for word in matched_words:
        print(f"Recognized Word: {word}")
    print(f"Sent detected objects to DataHub: {items_json}")

def send_image_to_datahub(color_image, edgeAgent):
    _, buffer = cv2.imencode('.jpg', color_image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    print("Sending Base64 data to DataHub")
    edgeData = EdgeData()
    tag = EdgeTag(deviceId='volume_camera', tagName='capture_image', value=jpg_as_text)
    edgeData.tagList.append(tag)
    edgeAgent.sendData(edgeData)
    print("Color image sent to DataHub from device volume_camera")

# TensorFlow 객체 인식 모델 및 레이블 로드
model_path = './saved_model'
label_map_path = './mscoco_label_map_food101.pbtxt'

# 파일을 읽고 내용 전달
with open(label_map_path, 'r', encoding='latin-1') as fid:
    label_map_string = fid.read()

label_map = string_int_label_map_pb2.StringIntLabelMap()
text_format.Merge(label_map_string, label_map)

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=101, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# SavedModel 로드 및 서명 확인
detection_model = tf.saved_model.load(model_path)
print(detection_model.signatures)

detection_fn = detection_model.signatures['serving_default']

def detect_fn(image):
    input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
    detections = detection_fn(input_1=input_tensor)
    return {key: value.numpy() for key, value in detections.items()}

# 웹캠 준비
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Webcam not detected. Exiting...")
    exit()

# 미리 저장된 단어
saved_words = ["pepsi", "25", "04", "02", "24", "09", "18"]

# 주요 프로세스 루프
last_time_sent = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_fn(frame)

    # 출력 키 확인
    print(detections.keys())

    # dense 출력의 내용을 확인하고 필요한 정보를 추출합니다.
    dense_output = detections['dense']
    print("Dense Output:", dense_output)

    # 여기서 dense_output의 형태를 파악하고, 필요한 정보(바운딩 박스, 클래스, 스코어 등)를 추출해야 합니다.
    # 만약 dense_output에 필요한 정보가 없다면, 모델이 잘못 훈련되었거나 저장되었을 수 있습니다.

    # 감지된 객체 정보 저장 (예제)
    detected_objects = []

    # 예제: dense_output에서 바운딩 박스와 클래스를 추출하는 방법
    # (실제 모델의 출력 형태에 따라 이 부분을 수정해야 합니다.)
    for i in range(dense_output.shape[0]):
        bbox = dense_output[i, :4]  # 바운딩 박스 좌표 (수정 필요)
        class_id = int(dense_output[i, 4])  # 클래스 ID (수정 필요)
        score = dense_output[i, 5]  # 점수 (수정 필요)
        if score > 0.50:
            class_name = category_index[class_id]['name']
            obj_id = str(uuid.uuid4())
            detected_objects.append({'id': obj_id, 'class_name': class_name, 'bbox': bbox.tolist(), 'score': score})

    # 텍스트 인식 및 매칭
    img_path = 'current_frame.png'
    cv2.imwrite(img_path, frame)
    result = infer(img_path)
    words_list = result['predictions'][0]['rec_texts']
    recognized_words = [re.sub(r'<[^>]*>', '', word) for word in words_list]
    matched_words = [word for word in recognized_words if word in saved_words]

    current_time = time.time()
    if current_time - last_time_sent >= 5:
        if recognized_words or matched_words:
            send_class_to_datahub(detected_objects)
            send_word_to_datahub(matched_words)
            #send_image_to_datahub(img_path, edgeAgent)
            last_time_sent = current_time

    # 바운딩 박스를 시각화하려면, dense_output에서 추출한 바운딩 박스와 클래스를 사용합니다.
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.array([obj['bbox'] for obj in detected_objects]),  # 바운딩 박스 좌표
        np.array([int(obj['class_name']) for obj in detected_objects]),  # 클래스 ID
        np.array([obj['score'] for obj in detected_objects]),  # 점수
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    cv2.imshow('Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time_to_wait = 5 - (current_time - last_time_sent)
    if time_to_wait > 0:
        time.sleep(time_to_wait)
cap.release()
cv2.destroyAllWindows()
