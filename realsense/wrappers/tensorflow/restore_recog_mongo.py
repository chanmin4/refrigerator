import os
import sys
import threading
import time
import json
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from scipy.spatial import distance
import requests
import uuid  # UUID import
import base64
from pymongo import MongoClient
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

# MongoDB 연결 설정
client = MongoClient('mongodb+srv://chanmin404:location1957!@refridge.g1vskrx.mongodb.net/')
db = client['smart_fridge']
volume_word_collection = db['Volume_word_Table']
internal_data_collection = db['InternalData']
total_volume_collection = db['total_volume']
captured_images_collection = db['captured_images']

def load_registered_objects():
    registered_objects = []
    # InternalData 컬렉션에서 모든 데이터를 조회
    data_cursor = internal_data_collection.find()
    for data in data_cursor:
        obj = {
            'external_id': data['external_id'],
            'object_class': data['object_class'],
            'last_seen': datetime.strptime(data['last_seen'], '%Y-%m-%d %H:%M:%S').timestamp(),
            'restore_flag': data['restore_flag'],
            'volume': data['volume'],
            'text': data['text']
        }
        registered_objects.append(obj)
    return registered_objects

@app.route('/collect', methods=['POST'])
def collect_data():
    data = request.json
    internal_data_collection.insert_one(data)
    return jsonify({"status": "success", "data": data}), 201

@app.route('/save_image', methods=['POST'])
def save_image():
    data = request.json
    image_data = {
        'timestamp': data['timestamp'],
        'image': base64.b64decode(data['image'])
    }
    db['captured_images'].update_one({}, {"$set": image_data}, upsert=True)
    return jsonify({"status": "success"}), 201

@app.route('/save_volume', methods=['POST'])
def save_volume():
    data = request.json
    volume_data = {
        'total_volume': data['total_volume'],
        'timestamp': data['timestamp']
    }
    db['total_volume'].update_one({}, {"$set": volume_data}, upsert=True)
    return jsonify({"status": "success"}), 201

@app.route('/get_internal_data', methods=['GET'])
def get_internal_data():
    data_cursor = internal_data_collection.find()
    data_list = []
    for data in data_cursor:
        data['_id'] = str(data['_id'])  # MongoDB ObjectId를 문자열로 변환
        data_list.append(data)
    return jsonify(data_list), 200

@app.route('/get_images', methods=['GET'])
def get_images():
    data_cursor = captured_images_collection.find()
    data_list = []
    for data in data_cursor:
        data['_id'] = str(data['_id'])  # MongoDB ObjectId를 문자열로 변환
        data['image'] = base64.b64encode(data['image']).decode('utf-8')  # 이미지를 base64로 인코딩
        data_list.append(data)
    return jsonify(data_list), 200

@app.route('/get_total_volume', methods=['GET'])
def get_total_volume():
    data_cursor = total_volume_collection.find()
    data_list = []
    for data in data_cursor:
        data['_id'] = str(data['_id'])  # MongoDB ObjectId를 문자열로 변환
        data_list.append(data)
    return jsonify(data_list), 200

def load_model_and_labels():
    model_path = './realsense/wrappers/tensorflow/frozen_inference_graph.pb'
    label_map_path = './realsense/wrappers/tensorflow/mscoco_label_map.pbtxt'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, 'rb') as fid:
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

            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(frame, axis=0)})

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

def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        return None
    return frame

def fetch_volume_word_data():
    return list(volume_word_collection.find())

def update_object_status(detected_objects, registered_objects, volume_text_data, category_index):
    current_time = time.time()
    used_ids = set(internal_data_collection.distinct('external_id'))
    new_objects = []

    for i in range(detected_objects['num_detections']):
        detected = {
            'class_name': category_index[detected_objects['detection_classes'][i]]['name'],
            'box': detected_objects['detection_boxes'][i],
            'score': detected_objects['detection_scores'][i]
        }
        matched = False
        
        for obj in registered_objects:
            if obj.get('object_class') == detected['class_name']:
                obj.update({'last_seen': current_time, 'restore_flag': False})
                matched = True
                break
        
        if not matched:
            for data_item in volume_text_data:
                external_id = data_item['id']
                if external_id not in used_ids:
                    detected.update({
                        'last_seen': current_time,
                        'restore_flag': False,
                        'store_food': detected['class_name'],
                        'volume': data_item['volume'],
                        'text': data_item['words'],
                        'external_id': external_id
                    })
                    insert_internal_data(external_id, detected['class_name'], current_time, False, data_item['volume'], data_item['words'])
                    new_objects.append(detected)
                    used_ids.add(external_id)
                    break

    registered_objects.extend(new_objects)
    objects_to_remove = []

    for obj in registered_objects[:]:
        last_seen_timestamp = obj['last_seen']
        if current_time - last_seen_timestamp > 10 and not obj['restore_flag']:
            obj['restore_flag'] = True
        if current_time - last_seen_timestamp > 60 and obj['restore_flag']:
            if 'object_class' in obj:
                internal_data_collection.delete_one({'external_id': obj['external_id']})
                volume_word_collection.delete_one({'external_id': obj['external_id']})
                print(f"Removed object with ID {obj['external_id']} food {obj['object_class']} from registered objects.")
            else:
                print(f"Warning: 'object_class' key is missing for object with ID {obj['external_id']}.")
            objects_to_remove.append(obj)
def insert_internal_data(external_id, object_class, last_seen, restore_flag, volume, text):
    formatted_last_seen = datetime.fromtimestamp(last_seen).strftime('%Y-%m-%d %H:%M:%S')
    
    data = {
        'external_id': external_id,
        'object_class': object_class,
        'last_seen': formatted_last_seen,
        'restore_flag': restore_flag,
        'volume': volume,
        'text': text
    }
    response = requests.post('http://localhost:5000/collect', json=data)
    if response.status_code == 201:
        print("Data inserted with last_seen formatted as:", formatted_last_seen)
    else:
        print("Failed to insert data.")

def save_total_volume(total_volume, timestamp):
    volume_data = {
        'total_volume': total_volume,
        'timestamp': timestamp
    }
    response = requests.post('http://localhost:5000/save_volume', json=volume_data)
    if response.status_code == 201:
        print("Total volume updated to MongoDB.")
    else:
        print("Failed to update total volume.")

def save_image(frame, timestamp):
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    image_data = {
        'timestamp': timestamp,
        'image': image_base64
    }
    
    response = requests.post('http://localhost:5000/save_image', json=image_data)
    if response.status_code == 201:
        print("Image saved to MongoDB.")
    else:
        print("Failed to save image.")

def run_detection():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Webcam not detected. Exiting...")
        return
    try:
        registered_objects = load_registered_objects()  # MongoDB에서 객체 정보를 로드
        model, category_index = load_model_and_labels()
        
        while True:
            frame = capture_frame(cap)
            if frame is None:
                continue
            detected_objects = detect_objects(frame, model, category_index)
            volume_text_data = fetch_volume_word_data()
            update_object_status(detected_objects, registered_objects, volume_text_data, category_index)
            
            print("Currently registered objects:")
            for obj in registered_objects:
                # 디버그용으로 obj 딕셔너리 출력
                print(f"Object: {obj}")

                # 'object_class'가 객체에 있는지 체크하고, 없는 경우를 처리
                if 'object_class' in obj:
                    last_seen_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(obj['last_seen']))
                    print(f"외부 ID: {obj.get('external_id', 'N/A')}, 클래스 이름: {obj['object_class']}, 마지막으로 본 시간: {last_seen_time}, 복구 플래그: {obj['restore_flag']}")
                else:
                    print("경고: 'object_class' 키가 객체에 없습니다.")
                    continue

            print("Detected objects in the current frame:")
            for i in range(detected_objects['num_detections']):
                print(f"Class: {category_index[detected_objects['detection_classes'][i]]['name']}, Score: {detected_objects['detection_scores'][i]}")
            current_time = time.time()
            current_time_str = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
            total_volume = sum(obj['volume'] for obj in registered_objects if 'volume' in obj)
            print(f"Total volume: {total_volume} L")
            # MongoDB에 이미지와 총 부피 저장
            save_image(frame, current_time_str)
            save_total_volume(total_volume, current_time_str)
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

def start_flask():
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    # Flask 서버를 별도의 스레드에서 실행
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.start()

    # run_detection 함수 실행
    run_detection()
