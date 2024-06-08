import cv2
import sys
sys.path.append('./mmocr_test')
from mmocr.apis import MMOCRInferencer
import re
import pyrealsense2 as rs
import numpy as np
import time
import json
import base64
import uuid
from collections import defaultdict
from realsense_device_manager import DeviceManager
from calibration_kabsch import PoseEstimation
from helper_functions import get_boundary_corners_2D
from measurement_task import cluster_and_bounding_box, calculate_boundingbox_points, calculate_cumulative_pointcloud, visualise_measurements
from sklearn.cluster import DBSCAN
from datahub_connector import edgeAgent
from wisepaasdatahubedgesdk.Model.Edge import EdgeData, EdgeTag
import matplotlib.pyplot as plt

def read_numbers_and_words_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    numbers_and_words = []
    for line in lines:
        line = line.strip()  # 줄 바꿈 문자 제거
        if line.isdigit():  # 숫자인 경우
            numbers_and_words.append(line)
        else:  # 단어인 경우
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
def send_data_to_datahub(volume, words, current_uuid):
    edgeData = EdgeData()
    data_payload = {
        'volume': volume,
        'words': words,
        'id': current_uuid
    }
    items_json = json.dumps(data_payload)
    tag = EdgeTag(deviceId="volume_camera", tagName="volume_and_words", value=items_json)
    edgeData.tagList.append(tag)
    edgeAgent.sendData(edgeData)
    print(f"Data sent to DataHub: {items_json}")
def generate_unique_id():
    return str(uuid.uuid4())

# 웹캠 캡처 및 텍스트 인식 함수
def capture_and_recognize(saved_words,current_uuid):
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
    final_volume=0
    while True:
        frames_devices = device_manager.poll_frames()
        

        point_cloud = calculate_cumulative_pointcloud(frames_devices, calibration_info_devices, roi_2D)
        clusters = cluster_and_bounding_box(point_cloud)
        bounding_box_points_color_image, length, width, height = calculate_boundingbox_points(clusters, calibration_info_devices)
        current_time = time.time()
    
        if len(clusters) == 1:
            volume = float(length) * float(width) * float(height)
            #height가 보정값으로 *10 되었으니 나누기1000000*10
            print(length,width,height/10)
            volume/=10000000
            
            if last_volume is not None:
                volume_change_ratio = abs(volume - last_volume) / last_volume * 100
                print(volume_change_ratio)
                if volume_change_ratio < 20:
                    if current_time - last_time_checked >= 15:
                        #이 사이클 횟수동안 돌린 부피를 평균부피로 계산해 정확도 상승기대
                        final_volume=volume
                        last_time_checked = current_time
                        visualise_measurements(frames_devices, bounding_box_points_color_image, length, width, height)    
                        break
            last_volume = volume

            visualise_measurements(frames_devices, bounding_box_points_color_image, length, width, height)      
        time.sleep(1)
    current_uuid=generate_unique_id()
    numbers_and_words = read_numbers_and_words_from_file('00.txt')   
    word_to_sent=capture_and_recognize(numbers_and_words,current_uuid)
    send_data_to_datahub(final_volume, word_to_sent, current_uuid)  # 글자가 인식되지 않은 경우 None으로 전송
    device_manager.disable_streams()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()
