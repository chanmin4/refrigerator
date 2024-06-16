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
from datetime import datetime,timezone, timedelta
ALLOWED_CLASSES = [44,52, 53, 54, 55, 56, 57, 58, 59, 61]
LABEL_SCORE_THRESHOLDS = {
    'banana': 0.3,
    'apple': 0.2,
    'sandwich': 0.3,
    'orange': 0.5,
    'broccoli': 0.3,
    'carrot': 0.3,
    'hot dog': 0.3,
    'pizza': 0.3,
    'cake': 0.1,
    'bottle':0.2
}
# MySQL 연결 설정
connection = pymysql.connect(
    host='smart-fridge.cn8m88cosddm.us-east-1.rds.amazonaws.com',
    user='chanmin4',
    password='location1957',
    db='smart-fridge',
    charset='utf8mb4',
    port=3307,
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
                'text': data['text']
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
    filtered_categories = [cat for cat in categories if cat['id'] in ALLOWED_CLASSES]
    category_index = label_map_util.create_category_index(filtered_categories)

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
                class_id = int(np.squeeze(output_dict['detection_classes'])[i])
                #score = float(np.squeeze(output_dict['detection_scores'])[i])
                #class_id = int(np.squeeze(output_dict['detection_classes'])[i])
                #if score > score_threshold and class_id in ALLOWED_CLASSES:
                    #detection_classes.append(class_id)
                    #detection_boxes.append(np.squeeze(output_dict['detection_boxes'])[i])
                    #detection_scores.append(score)
                # ALLOWED_CLASSES에 있는지 확인
                if class_id not in ALLOWED_CLASSES:
                    continue

                # category_index에 class_id가 있는지 확인
                if class_id not in category_index:
                    continue

                class_name = category_index[class_id]['name']
                
                if class_name in LABEL_SCORE_THRESHOLDS:
                    class_score_threshold = LABEL_SCORE_THRESHOLDS[class_name]
                else:
                    class_score_threshold = score_threshold

                if score > class_score_threshold and class_id in ALLOWED_CLASSES:
                    detection_classes.append(class_id)
                    detection_boxes.append(np.squeeze(output_dict['detection_boxes'])[i])
                    detection_scores.append(score)
            return {
                'num_detections': len(detection_classes),
                'detection_classes': np.array(detection_classes),
                'detection_boxes': np.array(detection_boxes),
                'detection_scores': np.array(detection_scores)
            }


def capture_frame(cap, resize_dim=(350,350), retries=5):
    for attempt in range(retries):
        try:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to capture frame on attempt {attempt + 1}")
                time.sleep(1)  # 잠시 대기 후 재시도
                continue
            frame = cv2.resize(frame, resize_dim)
            return frame
        except Exception as e:
            print(f"Error capturing frame on attempt {attempt + 1}: {e}")
            time.sleep(1)  # 잠시 대기 후 재시도
    return None

def fetch_volume_word_data():
    volume_text_data = []
    with connection.cursor() as cursor:
        sql = "SELECT * FROM volume_word_table"
        cursor.execute(sql)
        result = cursor.fetchall()
        for data in result:
                
                data['text'] = data['words']             
                volume_text_data.append(data)
    return volume_text_data

def update_object_status(detected_objects, registered_objects, volume_text_data, category_index):
    current_time = time.time()
    last_time=current_time
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
                obj.update({'last_seen': last_time, 'restore_flag': False})
                matched = True
                break
        if not matched:
            # 현재 탐지된 음식이 이미 등록된 음식인지 확인
            is_already_registered = False
            for obj in registered_objects:
                if obj.get('object_class') == detected['class_name']:
                    is_already_registered = True
                    break
            if not is_already_registered:
                for data_item in volume_text_data:
                    external_id = data_item['uuid']
                    if external_id not in used_ids:
                        current_time = time.time()
                        timestamp_col = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')                
                        detected.update({
                            'last_seen': last_time,
                            'restore_flag': False,
                            'store_food': detected['class_name'],
                            'volume': data_item['volume'],
                            'text': data_item['text'],
                            'external_id': external_id,
                            'timestamp_col':timestamp_col
                        })
                        insert_internal_data(external_id, detected['class_name'], last_time, False, data_item['volume'], data_item['text'],timestamp_col)
                        new_objects.append(detected)
                        used_ids.add(external_id)
                        break

    registered_objects.extend(new_objects)
    objects_to_remove = []

    for obj in registered_objects[:]:
        last_seen_timestamp = obj['last_seen']
        if current_time - last_seen_timestamp > 10 and not obj['restore_flag']:
            obj['restore_flag'] = 1
            with connection.cursor() as cursor:
                sql = "UPDATE internaldata SET restore_flag = %s WHERE external_id = %s"
                cursor.execute(sql, (obj['restore_flag'], obj['external_id']))
                connection.commit()
        if current_time - last_seen_timestamp > 120 and obj['restore_flag']:
            if 'object_class' in obj:
                with connection.cursor() as cursor:
                    sql = "DELETE FROM internaldata WHERE external_id = %s"
                    cursor.execute(sql, (obj['external_id'],))
                    sql = "DELETE FROM volume_word_table WHERE uuid = %s"
                    cursor.execute(sql, (obj['external_id'],))
                    connection.commit()
                print(f"Removed object with ID {obj['external_id']} food {obj['object_class']} from registered objects.")
            else:
                print(f"Warning: 'object_class' key is missing for object with ID {obj['external_id']}.")
            objects_to_remove.append(obj)

def insert_internal_data(external_id, object_class, last_seen, restore_flag, volume, text,timestamp_col):
    # UTC 시간으로부터 마지막으로 본 시간 (last_seen)을 생성
    last_seen_utc = datetime.fromtimestamp(last_seen)
    last_seen_korea = last_seen_utc
    formatted_last_seen = last_seen_korea.strftime('%Y-%m-%d %H:%M:%S')
    # timestamp_col이 문자열이라면 datetime 객체로 변환
    if isinstance(timestamp_col, str):
        timestamp_col = datetime.strptime(timestamp_col, '%Y-%m-%d %H:%M:%S')

    # expire_left 값을 계산
    expire_date_str = str(text)  # text는 yyyymmdd 형식의 날짜를 포함한다고 가정
    expire_date = datetime.strptime(expire_date_str, '%Y%m%d')
    # expire_date를 해당 날짜의 자정으로 설정
    expire_date = expire_date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    days_left = (expire_date - timestamp_col).days
    if object_class == 'banana':
        days_left=1
    if object_class == 'apple':
        days_left=-1
            

    with connection.cursor() as cursor:
        sql = "INSERT INTO internaldata (external_id, object_class, last_seen, restore_flag, volume, text,timestamp_col,expire_left) VALUES (%s, %s, %s, %s, %s, %s,%s,%s)"
        cursor.execute(sql, (external_id, object_class, formatted_last_seen, restore_flag, volume, text,timestamp_col, days_left))
        connection.commit()
    print("Data inserted with last_seen formatted as:", formatted_last_seen)

"""
def save_total_volume(total_volume, timestamp):
    with connection.cursor() as cursor:
        sql = "INSERT INTO total_volume (timestamp, total_volume) VALUES (%s, %s) ON DUPLICATE KEY UPDATE total_volume = VALUES(total_volume)"
        cursor.execute(sql, (timestamp, total_volume))
        connection.commit()
    print("Total volume updated to MySQL.")
    
"""
def save_total_volume(total_volume, timestamp):
    with connection.cursor() as cursor:
        sql = """
        INSERT INTO total_volume (timestamp, total_volume)
        VALUES (%s, %s)
        """
        cursor.execute(sql, (timestamp, total_volume))
        connection.commit()
    print("New total_volume inserted to MySQL.")
def save_image(frame, timestamp):
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    
    with connection.cursor() as cursor:
        sql = "INSERT INTO captured_images (timestamp, image) VALUES (%s, %s) ON DUPLICATE KEY UPDATE image = VALUES(image)"
        cursor.execute(sql, (timestamp, image_bytes))
        connection.commit()
    print("Image saved to MySQL.")

def run_detection():
    # DirectShow 백엔드를 사용하여 카메라 장치 열기
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    if not cap.isOpened():
        print("Webcam not detected. Exiting...")
        return

    frame_skip = 5  # 예: 5프레임마다 한 번씩 처리
    frame_count = 0
    try:
        registered_objects = load_registered_objects()  # MySQL에서 객체 정보를 로드
        model, category_index = load_model_and_labels()

        while True:
            frame_count += 1
            frame = capture_frame(cap, resize_dim=(350,350))
            if frame is None:
                print("Skipping frame due to capture failure.")
                continue
            if frame_count % frame_skip != 0:
                continue

            detected_objects = detect_objects(frame, model, category_index)
            volume_text_data = fetch_volume_word_data()
            update_object_status(detected_objects, registered_objects, volume_text_data, category_index)
            registered_objects = load_registered_objects()

            print("Currently registered objects:")
            for obj in registered_objects:
                print(f"Object: {obj}")
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
            save_image(frame, current_time)
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
            time.sleep(5)
    except Exception as e:
        print(f"Error during detection loop: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()