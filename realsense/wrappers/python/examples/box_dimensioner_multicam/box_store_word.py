import os
import sys
sys.path.append('./realsense/wrappers/python/examples/box_dimensioner_multicam/mmocr_test')
from mmocr.apis import MMOCRInferencer
import re
import threading
import time
import json
import cv2
import numpy as np
import tensorflow as tf
import pyrealsense2 as rs
#from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util
from scipy.spatial import distance
import requests
import uuid
import base64
from datetime import datetime
import pymysql
from collections import defaultdict
from realsense_device_manager import DeviceManager
from calibration_kabsch import PoseEstimation
from helper_functions import get_boundary_corners_2D
from measurement_task import cluster_and_bounding_box, calculate_boundingbox_points, calculate_cumulative_pointcloud, visualise_measurements
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
# 이미지 전처리 함수
def preprocess_image(image):
    # 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 명도 및 대비 조정
    enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
    # 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    # 이진화
    _, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
    return binary

# 유통기한 추출 함수
def extract_expiration_date(text):
    pattern = r'\b\d{4}\.\d{2}\.\d{2}\b'  # yyyy.mm.dd 형태
    matches = re.findall(pattern, text)
    return matches
# MySQL 연결 설정
mysql_connection = pymysql.connect(
    host='smart-fridge.cn8m88cosddm.us-east-1.rds.amazonaws.com',
    user='chanmin4',
    password='location1957',
    db='smart-fridge',
    charset='utf8mb4',
    port=3307,
    cursorclass=pymysql.cursors.DictCursor
)

def read_numbers_and_words_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    numbers_and_words = []
    for line in lines:
        line = line.strip()
        if line.isdigit():
            numbers_and_words.append(line)
        else:
            words = line.split()
            numbers_and_words.extend(words)

    return numbers_and_words

def visualize_calibration_status(frames, transformation_result_kabsch, intrinsics_devices, chessboard_params):
    for device_info, frame in frames.items():
        device = device_info[0]
        color_image = np.asanyarray(frame[rs.stream.color].get_data())
        pose_estimator = PoseEstimation(frames, intrinsics_devices, chessboard_params)
        ret, corners = pose_estimator.detect_chessboard(color_image)
        
        if ret:
            cv2.drawChessboardCorners(color_image, (chessboard_params[1], chessboard_params[0]), corners, ret)
        
        cv2.putText(color_image, f"Calibration {'Success' if transformation_result_kabsch[device][0] else 'Failure'}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow(f"Calibration - Device {device}", color_image)
        cv2.waitKey(1)

def send_data_to_mysql(volume, words, current_uuid):
  cursor = mysql_connection.cursor()
  sql = "INSERT INTO volume_word_table (volume, words, uuid) VALUES (%s, %s, %s)"
  # 단어 목록을 JSON 문자열로 변환
  words_json = json.dumps(words)

  # 데이터베이스에 삽입
  cursor.execute(sql, (volume, words_json, current_uuid))
  mysql_connection.commit()
  print(f"Data sent to MySQL: current_uuid={current_uuid}, volume={volume}, words={words}")
def generate_unique_id():
    cursor = mysql_connection.cursor()
    while True:
        new_uuid = str(uuid.uuid4())
        cursor.execute("SELECT 1 FROM volume_word_table WHERE uuid = %s", (new_uuid,))
        if cursor.fetchone() is None:
            return new_uuid
def capture_and_recognize(saved_words, current_uuid):
    print("Saved Words:", saved_words)
    infer = MMOCRInferencer(det='dbnetpp', rec='svtr-small')
    cap = cv2.VideoCapture(0)
    capture_interval = 5
    start_time = time.time()

    while True:
        current_time = time.time()
        ret, frame = cap.read()

        if not ret:
            break

        if current_time - start_time >= capture_interval:
            img_path = 'current_frame.png'
            cv2.imwrite(img_path, frame)
            result = infer(img_path)
            words_list = result['predictions'][0]['rec_texts']
            recognized_words = [re.sub(r'<[^>]*>', '', word) for word in words_list]
            matched_words = [word for word in recognized_words if word in saved_words]
            
            print("Recognized Words:", recognized_words)

            if matched_words:               
                print("Matched Words:", matched_words)
                return matched_words
            
            start_time = time.time()

        cv2.imshow('Webcam', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return []


def run_demo():
    current_uuid = None
    resolution_width = 640
    resolution_height = 360
    frame_rate = 30
    dispose_frames_for_stabilisation = 30
    chessboard_width = 6
    chessboard_height = 9
    square_size = 0.0253
    chessboard_size = (chessboard_width, chessboard_height)
    last_volume = None
    last_time_sent = time.time()
    last_time_checked = time.time()

    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
    rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
    rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

    device_manager = DeviceManager(rs.context(), rs_config)
    device_manager.enable_all_devices()
    for frame in range(dispose_frames_for_stabilisation):
        frames = device_manager.poll_frames()

    assert len(device_manager._available_devices) > 0
    intrinsics_devices = device_manager.get_device_intrinsics(frames)
    chessboard_params = [chessboard_height, chessboard_width, square_size]

    calibrated_device_count = 0
    while calibrated_device_count < len(device_manager._available_devices):
        frames = device_manager.poll_frames()
        pose_estimator = PoseEstimation(frames, intrinsics_devices, chessboard_params)
        transformation_result_kabsch = pose_estimator.perform_pose_estimation()
        object_point = pose_estimator.get_chessboard_corners_in3d()
        calibrated_device_count = 0
        for device_info in device_manager._available_devices:
            device = device_info[0]
            if not transformation_result_kabsch[device][0]:
                print("Place the chessboard on the plane where the object needs to be detected..")
            else:
                calibrated_device_count += 1

        visualize_calibration_status(frames, transformation_result_kabsch, intrinsics_devices, chessboard_params)

    transformation_devices = {}
    chessboard_points_cumulative_3d = np.array([-1, -1, -1]).transpose()
    for device_info in device_manager._available_devices:
        device = device_info[0]
        transformation_devices[device] = transformation_result_kabsch[device][1].inverse()
        points3D = object_point[device][2][:, object_point[device][3]]
        points3D = transformation_devices[device].apply_transformation(points3D)
        chessboard_points_cumulative_3d = np.column_stack((chessboard_points_cumulative_3d, points3D))

    chessboard_points_cumulative_3d = np.delete(chessboard_points_cumulative_3d, 0, 1)
    roi_2D = get_boundary_corners_2D(chessboard_points_cumulative_3d)

    print("Calibration completed... \nPlace the box in the field of view of the devices...")

    device_manager.enable_emitter(True)
    device_manager.load_settings_json("./HighResHighAccuracyPreset.json")
    extrinsics_devices = device_manager.get_depth_to_color_extrinsics(frames)
    calibration_info_devices = defaultdict(list)
    for calibration_info in (transformation_devices, intrinsics_devices, extrinsics_devices):
        for key, value in calibration_info.items():
            calibration_info_devices[key].append(value)
    while True:
        while True:
            frames_devices = device_manager.poll_frames()
            
            point_cloud = calculate_cumulative_pointcloud(frames_devices, calibration_info_devices, roi_2D)
            clusters = cluster_and_bounding_box(point_cloud)
            bounding_box_points_color_image, length, width, height = calculate_boundingbox_points(clusters, calibration_info_devices)
            current_time = time.time()
        
            if len(clusters) == 1:
                volume = float(length) * float(width) * float(height)
                # height가 보정값으로 *10 되었으니 나누기1000000*10
                print(length, width, height / 10)
                volume /= 10000000
                print(volume)
                """
                부피 변화율은 현재 측정된 부피와 이전 측정된 부피를 비교하여 계산
                recent_volumes[-2]를 사용하는 것은 현재 측정된 부피를 
                recent_volumes[-1], 이전 측정된 부피를 recent_volumes[-2]으로 해서
                """
                # 최근 5번 측정된 부피 기록 (초기에는 5번 채워짐)
                if 'recent_volumes' not in locals():
                    recent_volumes = []
                if last_volume is not None:
                    recent_volumes.append(last_volume)
                if len(recent_volumes) >= 5:
                    recent_volumes.pop(0)  # 첫번째 값 제거
                recent_volumes.append(volume)
                # 최근 부피 변화율 계산 (최근 5번 기준, 누적 평균 사용)
                if len(recent_volumes) >= 5:
                    # 누적 평균 계산
                    average_volume = sum(recent_volumes) / len(recent_volumes)
                    # 최근 5번째 측정된 부피
                    previous_volume = recent_volumes[-5]
                    volume_change_ratio = abs(volume - previous_volume) / average_volume * 100
                    print(f"최근5번 부피 평균변화량:{volume_change_ratio}%")
                    # 최근 부피 변화율이 20% 미만이면 다음 로직 수행
                    if volume_change_ratio < 25:
                            # 이 사이클 횟수동안 돌린 부피를 평균부피로 계산해 정확도 상승기대
                            last_volume = volume
                            last_time_checked = current_time
                            visualise_measurements(frames_devices, bounding_box_points_color_image, length, width, height)    
                            break
                last_volume = volume

                visualise_measurements(frames_devices, bounding_box_points_color_image, length, width, height)      
            time.sleep(2)

        current_uuid = generate_unique_id()
        numbers_and_words = read_numbers_and_words_from_file('00.txt')   
        word_to_sent = capture_and_recognize(numbers_and_words, current_uuid)
        send_data_to_mysql(last_volume, word_to_sent, current_uuid)  # 글자가 인식되지 않은 경우 None으로 전송
        device_manager.disable_streams()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()
