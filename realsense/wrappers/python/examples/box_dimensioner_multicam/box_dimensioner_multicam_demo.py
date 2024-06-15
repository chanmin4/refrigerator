
import sys
sys.path.append('./realsense/wrappers/python/examples/box_dimensioner_multicam/mmocr_test')
from mmocr.apis import MMOCRInferencer
import re
import time
import json
import cv2
import numpy as np
import tensorflow as tf
import pyrealsense2 as rs
#from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util
from scipy.spatial import distance
import uuid
from datetime import datetime,timezone, timedelta
import pymysql
from collections import defaultdict
from realsense_device_manager import DeviceManager
from calibration_kabsch import PoseEstimation
from helper_functions import get_boundary_corners_2D
from measurement_task import cluster_and_bounding_box, calculate_boundingbox_points, calculate_cumulative_pointcloud, visualise_measurements
import matplotlib.pyplot as plt
# 이미지 전처리 함수
start_key = ord('s')  # 's' 키를 누르면 시작
def preprocess_image(image):
  # 흑백 변환 유지 (필요)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(image)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    thresh = cv2.adaptiveThreshold(opened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

# 유통기한 추출 함수
def extract_expiration_date(text):
    pattern = r'\b\d{8}\b'  # yyyymmdd 형태
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

def save_total_volume(total_volume, timestamp):
    cursor = mysql_connection.cursor()
    sql = """
    INSERT INTO total_volume (timestamp, total_volume)
    VALUES (%s, %s)
    """
    cursor.execute(sql, (timestamp, total_volume))
    mysql_connection.commit()
    print("New total_volume inserted to MySQL.")
def send_data_to_mysql(volume, words, current_uuid):
  cursor = mysql_connection.cursor()
  sql = "INSERT INTO volume_word_table (volume, words, uuid) VALUES (%s, %s, %s)"

  # 데이터베이스에 삽입
  cursor.execute(sql, (volume, words, current_uuid))
  mysql_connection.commit()
  print(f"Data sent to MySQL: current_uuid={current_uuid}, volume={volume}, words={words}")
def generate_unique_id():
    cursor = mysql_connection.cursor()
    while True:
        new_uuid = str(uuid.uuid4())
        cursor.execute("SELECT 1 FROM volume_word_table WHERE uuid = %s", (new_uuid,))
        if cursor.fetchone() is None:
            return new_uuid
def calculate_total_volume():
    cursor = mysql_connection.cursor()
    cursor.execute("SELECT total_volume FROM total_volume ORDER BY timestamp DESC LIMIT 1")
    result = cursor.fetchone()
    if result and result['total_volume'] is not None:
        return float(result['total_volume'])
    else:
        return 0.0     
def insert_internal_data(external_id, object_class, last_seen, restore_flag, volume, text, timestamp_col):
    last_seen_utc = datetime.fromtimestamp(last_seen)
    last_seen_korea = last_seen_utc
    formatted_last_seen = last_seen_korea.strftime('%Y-%m-%d %H:%M:%S')
    
    if isinstance(timestamp_col, str):
        timestamp_col = datetime.strptime(timestamp_col, '%Y-%m-%d %H:%M:%S')

    expire_date_str = str(text)
    expire_date = datetime.strptime(expire_date_str, '%Y%m%d')
    expire_date = expire_date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    days_left = (expire_date - timestamp_col).days

    with mysql_connection.cursor() as cursor:
        sql = "INSERT INTO internaldata (external_id, object_class, last_seen, restore_flag, volume, text, timestamp_col, expire_left) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        cursor.execute(sql, (external_id, object_class, formatted_last_seen, restore_flag, volume, text, timestamp_col, days_left))
        mysql_connection.commit()
    print("Data inserted with last_seen formatted as:", formatted_last_seen)

def capture_and_recognize(volume, current_uuid):
    infer = MMOCRInferencer(det='dbnetpp', rec='svtr-small')
    cap = cv2.VideoCapture(0)
    capture_interval = 4
    start_time = time.time()

    while True:
        current_time = time.time()
        ret, frame = cap.read()

        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x = int(frame.shape[1] / 2) - 50
        y = int(frame.shape[0] / 2) - 50
        w = 450
        h = 450
        cropped_img = gray[y:y+h, x:x+w]
        preprocessed_frame = preprocess_image(cropped_img)
        if current_time - start_time >= capture_interval:
            img_path = 'current_frame.png'
            cv2.imwrite(img_path, frame)
            result = infer(img_path)
            words_list = result['predictions'][0]['rec_texts']
            recognized_words = [re.sub(r'<[^>]*>', '', word) for word in words_list]
            expiration_dates = extract_expiration_date(' '.join(recognized_words))
            print("Recognized Words:", recognized_words)
            
            word_to_sent = recognized_words
            detected_item = None
            
            # Oreo 또는 Ritz가 인식된 경우
            for word in recognized_words:
                if word.lower() in ['oreo', 'ritz'] and not re.match(r'\b\d{8}\b', word):
                    print(f"Detected: {word}")
                    detected_item = word.lower()
                    break
            
            if detected_item:
                # Oreo 또는 Ritz가 인식되면 유통기한을 추가로 탐지
                start_time = time.time()
                while True:
                    current_time = time.time()
                    ret, frame = cap.read()

                    if not ret:
                        break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    x = int(frame.shape[1] / 2) - 50
                    y = int(frame.shape[0] / 2) - 50
                    w = 450
                    h = 450
                    cropped_img = gray[y:y+h, x:x+w]
                    preprocessed_frame = preprocess_image(cropped_img)
                    if current_time - start_time >= capture_interval:
                        img_path = 'current_frame.png'
                        cv2.imwrite(img_path, frame)
                        result = infer(img_path)
                        words_list = result['predictions'][0]['rec_texts']
                        recognized_words = [re.sub(r'<[^>]*>', '', word) for word in words_list]
                        expiration_dates = extract_expiration_date(' '.join(recognized_words))
                        print("Recognized Words (Second Detection):", recognized_words)

                        if expiration_dates:
                            word_to_sent = expiration_dates[0] # 이름과 유통기한 설정
                        else:
                            expiration_dates = None
                        timestamp_col = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                        
                        if expiration_dates:
                            
                            print("Expiration Dates:", expiration_dates)
                            insert_internal_data(current_uuid, detected_item, current_time, 0, volume,word_to_sent , timestamp_col)
                            total_volume = calculate_total_volume()
                            current_datetime = datetime.fromtimestamp(current_time)
                            timestamp_col = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
                            save_total_volume(total_volume+volume,timestamp_col)
                            return word_to_sent
                        
                        start_time = time.time()
            else:
                # 유통기한이 인식된 경우 탐지 종료x
                if expiration_dates:
                    word_to_sent = expiration_dates
                    return word_to_sent
           
            
        cv2.imshow('Webcam', preprocessed_frame)
        cv2.waitKey(1)

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
    running = False  # 초기 상태에서는 비활성화
    #cap = cv2.VideoCapture(0)  
    while True:
        black_image = np.zeros((240, 320, 3), dtype=np.uint8)
        black_image.fill(0)  # Fill the image with black color

        # Create a window named "Black Window"
        window_name = "Black Window"
        cv2.namedWindow(window_name)
        
# Display the black image in the window
        # 사용자 입력 확인
        key = cv2.pollKey()
        if key != -1:
            # 's' 키 입력 시 시작
            if key == start_key:
                running = True
                print("측정 시작")
                cv2.destroyWindow("Black Window")

            # 'q' 키 입력 시 종료
            elif key == ord('q'):
                break
        else: 
            continue
            
            
        
        time.sleep(1)
        if running:
            while running:
                cv2.waitKey(1)
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
                    if volume is not None:
                        recent_volumes.append(volume)
                    if len(recent_volumes) >= 6:
                        recent_volumes.pop(0)  # 첫번째 값 제거
                    # 최근 부피 변화율 계산 (최근 5번 기준, 누적 평균 사용)
                    if len(recent_volumes) >= 5:
                        # 누적 평균 계산
                        average_volume = sum(recent_volumes) / len(recent_volumes)
                        # 최근 5번째 측정된 부피
                        previous_volume = recent_volumes[-5]
                        volume_change_ratio = abs(volume - previous_volume) / average_volume * 100
                        print(f"최근5번 측정 부피 평균의 변화량:{volume_change_ratio}%")
                        # 최근 부피 변화율이 20% 미만이면 다음 로직 수행
                        if volume_change_ratio < 20:
                                # 이 사이클 횟수동안 돌린 부피를 평균부피로 계산해 정확도 상승기대
                                last_volume = volume
                                last_time_checked = current_time
                                visualise_measurements(frames_devices, bounding_box_points_color_image, length, width, height)    
                                break
                    last_volume = volume

                    
                    visualise_measurements(frames_devices, bounding_box_points_color_image, length, width, height)      
                
                time.sleep(2)
            #ret, frame = cap.read()
            current_uuid = generate_unique_id()
            #numbers_and_words = read_numbers_and_words_from_file('00.txt')  
            #현재문제점( 두번째  while부터 이거때매 cv2 고정됨) 
            word_to_sent = capture_and_recognize(last_volume,current_uuid)  
            send_data_to_mysql(last_volume, word_to_sent, current_uuid)  # 글자가 인식되지 않은 경우 None으로 전송
            running=False
            #device_manager.disable_streams()
            recent_volumes = []
    #cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    run_demo()
