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
    # 각 id별 클래스 이름을 출력
    for item in detected_items:
        
        print(f"Class Name:{item['class_name']}")
    
    print(f"Sent detected objects to DataHub: {items_json}")
def send_word_to_datahub(matched_words):
    edgeData = EdgeData()
    items_json = json.dumps(matched_words)
    tag = EdgeTag(deviceId="volume_camera", tagName="words", value=items_json)
    edgeData.tagList.append(tag)
    edgeAgent.sendData(edgeData)
    # 글자출력
    for word in matched_words:
        print(f"Recognized Word: {word}")  # 단어를 직접 출력
    print(f"Sent detected objects to DataHub: {items_json}")
def send_image_to_datahub(color_image, edgeAgent):
    # 이미지를 Base64로 인코딩하여 전송
    _, buffer = cv2.imencode('.jpg', color_image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    
    # Debug: Print the Base64 data before sending
    print("Sending Base64 data to DataHub")
    
    edgeData = EdgeData()
    tag = EdgeTag(deviceId='volume_camera', tagName='capture_image', value=jpg_as_text)
    edgeData.tagList.append(tag)
    edgeAgent.sendData(edgeData)
    print("Color image sent to DataHub from device volume_camera")
# TensorFlow 객체 인식 모델 및 레이블 로드
model_path = './frozen_inference_graph.pb'
label_map_path = './mscoco_label_map.pbtxt'
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# 웹캠 준비
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Webcam not detected. Exiting...")
    exit()

# 미리 저장된 단어
saved_words = ["pepsi", "25", "04", "02", "24", "09", "18"]

# 주요 프로세스 루프
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        tensor_dict = {key: detection_graph.get_tensor_by_name(key + ':0') for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']}
        last_time_sent = time.time()
        while True:
            
            ret, frame = cap.read()
            if not ret:
                break

            image_np_expanded = np.expand_dims(frame, axis=0)
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})

            # 객체 인식 정보 추출 및 출력
            detected_objects = []
            for i in range(int(output_dict['num_detections'][0])):
                score = output_dict['detection_scores'][0][i]
                if score > 0.50:
                    class_id = int(output_dict['detection_classes'][0][i])
                    class_name = category_index[class_id]['name']
                    obj_id = str(uuid.uuid4())
                    detected_objects.append({'id': obj_id, 'class_name': class_name})

            # 텍스트 인식 및 매칭
            img_path = 'current_frame.png'
            cv2.imwrite(img_path, frame)
            result = infer(img_path)
            words_list = result['predictions'][0]['rec_texts']
            recognized_words = [re.sub(r'<[^>]*>', '', word) for word in words_list]
            matched_words = [word for word in recognized_words if word in saved_words]

            current_time = time.time()
            #이거 수정 if문 거치고 **맨마지막에 대기하도록해야함 안그럼 전송은 안보내는데 계속 while돔
            #여기서 이미지 보내도록 수정필요 
            if current_time - last_time_sent >= 5:
                if recognized_words or matched_words:
                    send_class_to_datahub(detected_objects)
                    send_word_to_datahub(matched_words)
                    #send_image_to_datahub(img_path, edgeAgent)
                    last_time_sent = current_time
                     
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(output_dict['detection_boxes']),
                np.squeeze(output_dict['detection_classes']).astype(np.int32),
                np.squeeze(output_dict['detection_scores']),
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
