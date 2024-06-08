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
import uuid
import base64
import pymysql
from datetime import datetime

# MySQL 연결 설정
connection = pymysql.connect(
    host='localhost',
    user='chanmin4',
    password='location1957',
    db='smart_fridge',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

def load_registered_objects():
    registered_objects = []
    with connection.cursor() as cursor:
        sql = "SELECT * FROM internaldata"
        cursor.execute(sql)
        result = cursor.fetchall()
        for data in result:
            obj = {
                'external_id': data['external_id'],
                'object_class': data['object_class'],
                'last_seen': data['last_seen'].timestamp(),
                'restore_flag': data['restore_flag'],
                'volume': data['volume'],
                'text': json.loads(data['text'])
            }
            registered_objects.append(obj)
    return registered_objects

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
    volume_text_data = []
    with connection.cursor() as cursor:
        sql = "SELECT * FROM volume_word_table"
        cursor.execute(sql)
        result = cursor.fetchall()
        for data in result:
            try:
                # 데이터가 JSON 형식인지 확인
                data['text'] = json.loads(data['words'])
            except json.JSONDecodeError:
                # JSON 형식이 아니면 쉼표로 구분된 문자열로 처리
                data['text'] = data['words'].split(',')
            volume_text_data.append(data)
    return volume_text_data

def update_object_status(detected_objects, registered_objects, volume_text_data, category_index):
    current_time = time.time()
    used_ids = set()
    with connection.cursor() as cursor:
        sql = "SELECT external_id FROM internaldata"
        cursor.execute(sql)
        result = cursor.fetchall()
        for data in result:
            used_ids.add(data['external_id'])

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
                        'text': data_item['text'],
                        'external_id': external_id
                    })
                    insert_internal_data(external_id, detected['class_name'], current_time, False, data_item['volume'], data_item['text'])
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
                with connection.cursor() as cursor:
                    sql = "DELETE FROM InternalData WHERE external_id = %s"
                    cursor.execute(sql, (obj['external_id'],))
                    sql = "DELETE FROM Volume_word_Table WHERE id = %s"
                    cursor.execute(sql, (obj['external_id'],))
                    connection.commit()
                print(f"Removed object with ID {obj['external_id']} food {obj['object_class']} from registered objects.")
            else:
                print(f"Warning: 'object_class' key is missing for object with ID {obj['external_id']}.")
            objects_to_remove.append(obj)

def insert_internal_data(external_id, object_class, last_seen, restore_flag, volume, text):
    formatted_last_seen = datetime.fromtimestamp(last_seen).strftime('%Y-%m-%d %H:%M:%S')
    
    with connection.cursor() as cursor:
        sql = "INSERT INTO InternalData (external_id, object_class, last_seen, restore_flag, volume, text) VALUES (%s, %s, %s, %s, %s, %s)"
        cursor.execute(sql, (external_id, object_class, formatted_last_seen, restore_flag, volume, json.dumps(text)))
        connection.commit()
    print("Data inserted with last_seen formatted as:", formatted_last_seen)

def save_total_volume(total_volume, timestamp):
    with connection.cursor() as cursor:
        sql = "INSERT INTO total_volume (timestamp, total_volume) VALUES (%s, %s) ON DUPLICATE KEY UPDATE total_volume = VALUES(total_volume)"
        cursor.execute(sql, (timestamp, total_volume))
        connection.commit()
    print("Total volume updated to MySQL.")

def save_image(frame, timestamp):
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    
    with connection.cursor() as cursor:
        sql = "INSERT INTO captured_images (timestamp, image) VALUES (%s, %s) ON DUPLICATE KEY UPDATE image = VALUES(image)"
        cursor.execute(sql, (timestamp, image_bytes))
        connection.commit()
    print("Image saved to MySQL.")

def run_detection():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Webcam not detected. Exiting...")
        return
    try:
        registered_objects = load_registered_objects()  # MySQL에서 객체 정보를 로드
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
            # MySQL에 이미지와 총 부피 저장
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

if __name__ == "__main__":
    # run_detection 함수 실행
    run_detection()
