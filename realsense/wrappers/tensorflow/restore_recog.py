import os
import sys
path = "./realsense/wrappers/tensorflow"
sys.path.append(os.path.abspath(path))
import time
import json
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from scipy.spatial import distance
from wisepaasdatahubedgesdk.EdgeAgent import EdgeAgent
from wisepaasdatahubedgesdk.Model.Edge import EdgeAgentOptions, DCCSOptions, EdgeData, EdgeTag, constant
import requests
import uuid  # UUID import
import base64
# EdgeAgent 설정 및 연결
edgeAgentOptions = EdgeAgentOptions(nodeId='3607ae3d-5e1e-4171-8706-b6a111fa05ac',
                                    connectType=constant.ConnectType['DCCS'],
                                    DCCS=DCCSOptions(apiUrl='https://api-dccs-ensaas.sa.wise-paas.com/',
                                                     credentialKey='8d47cc1fab2e0a5207ab7da336ae4atl'))
edgeAgent = EdgeAgent(edgeAgentOptions)
edgeAgent.connect()
def send_image_to_datahub(color_image, edgeAgent):
    _, buffer = cv2.imencode('.jpg', color_image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    
    print("Sending Base64 data to DataHub")
    
    edgeData = EdgeData()
    tag = EdgeTag(deviceId='volume_camera', tagName='capture_image', value=jpg_as_text)
    edgeData.tagList.append(tag)
    edgeAgent.sendData(edgeData)
    print("Color image sent to DataHub from device volume_camera")
def send_data_to_datahub(tag_name, data, edgeAgent):
    edgeData = EdgeData()
    items_json = json.dumps(data)
    tag = EdgeTag(deviceId="volume_camera", tagName=tag_name, value=items_json)
    edgeData.tagList.append(tag)
    edgeAgent.sendData(edgeData)
    print(f"Data sent to DataHub: {items_json}")

def send_total_volume_to_datahub(total_volume, edgeAgent):
    
    
    # 데이터허브로 전송
    edgeData = EdgeData()
    tag = EdgeTag(deviceId="volume_camera", tagName="total_volume", value=total_volume)
    edgeData.tagList.append(tag)
    edgeAgent.sendData(edgeData)
    print(f"Sent total volume to DataHub: {total_volume} L")
def send_refridge_volume_to_datahub(cal_storage, edgeAgent):  
    # 데이터허브로 전송
    edgeData = EdgeData()
    tag = EdgeTag(deviceId="volume_camera", tagName="cal_storage", value=cal_storage)
    edgeData.tagList.append(tag)
    edgeAgent.sendData(edgeData)
    print(f"Sent total volume to DataHub: {cal_storage} L")
def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        return None
    return frame

def load_model_and_labels():
    model_path = './realsense/wrappers/tensorflow/frozen_inference_graph.pb'
    label_map_path = './realsense/wrappers/tensorflow/mscoco_label_map.pbtxt'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        # 파일을 바이너리 모드로 열기
        with tf.io.gfile.GFile(model_path, 'rb') as fid:  # 'rb' 모드 추가
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return detection_graph, category_index

def detect_objects(frame, detection_graph, category_index, score_threshold=0.5):
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            tensor_dict = {
                'num_detections': detection_graph.get_tensor_by_name('num_detections:0'),
                'detection_boxes': detection_graph.get_tensor_by_name('detection_boxes:0'),
                'detection_scores': detection_graph.get_tensor_by_name('detection_scores:0'),
                'detection_classes': detection_graph.get_tensor_by_name('detection_classes:0')
            }

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(frame, axis=0)})

            # Filter results by score_threshold
            num_detections = int(output_dict['num_detections'][0])
            detection_classes = []
            detection_boxes = []
            detection_scores = []
            for i in range(num_detections):
                score = float(np.squeeze(output_dict['detection_scores'])[i])
                if score > score_threshold:
                    detection_classes.append(int(np.squeeze(output_dict['detection_classes'])[i]))
                    detection_boxes.append(np.squeeze(output_dict['detection_boxes'])[i])
                    detection_scores.append(score)

            return {
                'num_detections': len(detection_classes),
                'detection_classes': np.array(detection_classes),
                'detection_boxes': np.array(detection_boxes),
                'detection_scores': np.array(detection_scores)
            }
def update_data_in_datahub( volume_text_data,exclude_id):
    access_token = get_access_token()
    if access_token:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        updated_data = [item for item in volume_text_data if item[2] != exclude_id]
        # 데이터 업데이트
        update_url = f"https://portal-datahub-trainingapps-eks004.sa.wise-paas.com/v1/SimpleJsons/setValue"
        update_payload = {
            "nodeId": "3607ae3d-5e1e-4171-8706-b6a111fa05ac",
            "deviceId": "volume_camera",
            "tagName": "volume_and_words",
            "value": json.dumps(updated_data)
    }
        update_response = requests.put(update_url, headers=headers, json=update_payload)
        if update_response.status_code == 200:
            print("Data updated successfully in DataHub.")
        else:
            print("Failed to update data in DataHub:", update_response.status_code, update_response.text)
    else:
        print("Access token could not be retrieved.")

def cleanup_resources(cap, sess):
    cap.release()
    cv2.destroyAllWindows()
    sess.close()
# API에서 액세스 토큰을 가져오는 함수
def get_access_token():
    url = "https://api-sso-ensaas.sa.wise-paas.com/v4.0/auth/native"
    data = {
        "username": "chanmin4@naver.com", 
        "password": "Location1957!",      
        "userDetail": True
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # 요청 실패 시 예외 발생

        # 응답 JSON을 파싱하여 액세스 토큰 가져오기
        result = response.json()
        access_token = result.get('accessToken')
        return access_token

    except requests.exceptions.RequestException as e:
        print("Failed to get access token:", e)
        return None
# 외부측 데이터 가져오기
def fetch_volume_and_text_data():
    access_token = get_access_token()
    if access_token:
        url = "https://portal-datahub-trainingapps-eks004.sa.wise-paas.com/api/v1/RealData/raw"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {access_token}"  # 액세스 토큰 포함
        }
        payload = json.dumps([
            {
                "nodeId": "3607ae3d-5e1e-4171-8706-b6a111fa05ac",
                "deviceId": "volume_camera",
                "tagName": "volume_and_words"
            }
        ])
            
        response = requests.post(url, headers=headers, data=payload)
            
        if response.status_code == 200:
            data = response.json()
            print("Data fetched successfully:", data)
            return data
        else:
            print("Failed to fetch data:", response.status_code)
            return []
    else:
        print("Access token could not be retrieved.")
        return []
def update_object_status(detected_objects, registered_objects, volume_text_data, category_index):
    current_time = time.time()
    used_ids = {obj.get('external_id') for obj in registered_objects if 'external_id' in obj}  # 사용된 외부 ID 리스트
    new_objects = []

    for i in range(detected_objects['num_detections']):
        detected = {
            'class_name': category_index[detected_objects['detection_classes'][i]]['name'],
            'box': detected_objects['detection_boxes'][i],
            'score': detected_objects['detection_scores'][i]
        }
        matched = False
        
        for obj in registered_objects:
            if obj['store_food'] == detected['class_name']:
                obj.update({'last_seen': current_time, 'restore_flag': False})
                matched = True
        # 한번에 같은 종류 등록되는거 생각필요 ex)물 , 물 동시등록
        if not matched:
            for data_item in volume_text_data:
                external_id = data_item[2]  # 외부 ID 가져오기
                if external_id not in used_ids:
                    detected.update({
                        'last_seen': current_time,
                        #flag false일시 물체 냉장고안존재상태 (복구아님) true면 갑자기 10초이상동안
                        #음식이 사라져서 감지를 못하고 삭제할지 결정 유예기간상태
                        'restore_flag': False,
                        'store_food': detected['class_name'],
                        'volume': data_item[0],
                        'text': data_item[1],
                        'external_id': external_id
                    })
                    new_objects.append(detected)
                    used_ids.add(external_id)
                    break

    registered_objects.extend(new_objects)
    objects_to_remove = []  
    # 인식 중단된 객체 처리
    for obj in registered_objects[:]:
        if current_time - obj['last_seen'] > 10 and not obj['restore_flag']:  # 10초 이상 업데이트되지 않은 경우
            obj['restore_flag'] = True
        if current_time - obj['last_seen'] > 60 and obj['restore_flag']:
            update_data_in_datahub(volume_text_data,obj['external_id'])
            objects_to_remove.append(obj)# 60초 이상 경과한 경우 삭제
            registered_objects.remove(obj)
            print(f"Removed object with ID {obj['exteprnal_id']} from registered objects.")
        
        
def run_detection():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Webcam not detected. Exiting...")
        return
    try:
        registered_objects = []
        model, category_index = load_model_and_labels()  # 모델과 라벨 로딩
        
        while True:
            frame = capture_frame(cap)  # 프레임 캡처 함수
            if frame is None:
                continue
            send_image_to_datahub(frame, edgeAgent)
            detected_objects = detect_objects(frame, model, category_index)  # 객체 탐지 함수
            volume_text_data = fetch_volume_and_text_data()  # 부피와 글자 정보 조회
            update_object_status(detected_objects, registered_objects, volume_text_data, category_index)
            
            # Debug: 출력 객체 상태 및 감지된 객체 정보
            print("Currently registered objects:")
            for obj in registered_objects:
                last_seen_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(obj['last_seen']))
                print(f"External ID: {obj.get('external_id', 'N/A')}, Class Name: {obj['class_name']}, Last Seen: {last_seen_time}, Restore Flag: {obj['restore_flag']}")

            print("Detected objects in the current frame:")
            for i in range(detected_objects['num_detections']):
                print(f"Class: {category_index[detected_objects['detection_classes'][i]]['name']}, Score: {detected_objects['detection_scores'][i]}")

            object_classes = [obj['class_name'] for obj in registered_objects]
            send_data_to_datahub("internal_data", object_classes, edgeAgent)  # 데이터 허브로 상태 업데이트 전송
            
            total_volume = 0.0
            for obj in registered_objects:
                # 부피 데이터가 숫자로만 주어진 경우 바로 float 타입으로 변환합니다.
                try:
                    volume = float(obj['volume'])
                except ValueError:
                    print(f"Invalid volume format: {obj['volume']}")
                    continue  # 형식이 올바르지 않으면 이 객체는 건너뜁니다.
                total_volume += volume
            send_total_volume_to_datahub(total_volume, edgeAgent)  # 전체 부피 데이터 허브로 전송
            cal_storage=total_volume/(24.4)*100
            
            send_refridge_volume_to_datahub(cal_storage, edgeAgent)
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                detected_objects['detection_boxes'],
                detected_objects['detection_classes'].astype(np.int32),
                detected_objects['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            cv2.imshow('Detection Results', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(1)    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()