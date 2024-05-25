import os
import sys
import pyrealsense2 as rs
import numpy as np
import tensorflow as tf
import cv2
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from collections import defaultdict
from sklearn.cluster import DBSCAN

# 현재 스크립트의 디렉터리를 기준으로 상대 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'wrappers', 'python', 'examples', 'box_dimensioner_multicam'))

# 모듈 임포트
from realsense_device_manager import DeviceManager
from calibration_kabsch import PoseEstimation
from helper_functions import get_boundary_corners_2D
from measurement_task import cluster_pointcloud, calculate_boundingbox_points, calculate_cumulative_pointcloud, visualise_measurements

# Object Detection 설정
PATH_TO_CKPT = os.path.join(current_dir, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(current_dir, 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Define some constants
resolution_width = 1280  # pixels
resolution_height = 720  # pixels
frame_rate = 30  # fps

dispose_frames_for_stabilisation = 30  # frames

chessboard_width = 6  # squares
chessboard_height = 9  # squares
square_size = 0.0253  # meters

# RealSense camera setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)
config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)

pipeline_started = False  # 파이프라인이 시작되었는지 여부를 추적

def start_pipeline():
    global pipeline_started
    if pipeline_started:
        pipeline.stop()  # 기존 스트림 중지
        pipeline_started = False
    pipeline.start(config)
    pipeline_started = True  # 파이프라인 시작됨
    # Allow camera to stabilize
    time.sleep(2)  # Sleep for 2 seconds to let the camera stabilize
    # Ignore initial frames for auto-exposure
    for _ in range(dispose_frames_for_stabilisation):
        pipeline.wait_for_frames()

start_pipeline()

# Use the device manager class to enable the devices and get the frames
device_manager = DeviceManager(rs.context(), config)
device_manager.enable_all_devices()

# Allow some frames for the auto-exposure controller to stabilise
for frame in range(dispose_frames_for_stabilisation):
    frames = device_manager.poll_frames()

assert len(device_manager._available_devices) > 0

# Calibration
# Get the intrinsics of the realsense device
intrinsics_devices = device_manager.get_device_intrinsics(frames)

# Set the chessboard parameters for calibration
chessboard_params = [chessboard_height, chessboard_width, square_size]

# Estimate the pose of the chessboard in the world coordinate using the Kabsch Method
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

# Save the transformation object for all devices in an array to use for measurements
transformation_devices = {}
chessboard_points_cumulative_3d = np.array([-1, -1, -1]).transpose()
for device_info in device_manager._available_devices:
    device = device_info[0]
    transformation_devices[device] = transformation_result_kabsch[device][1].inverse()
    points3D = object_point[device][2][:, object_point[device][3]]
    points3D = transformation_devices[device].apply_transformation(points3D)
    chessboard_points_cumulative_3d = np.column_stack((chessboard_points_cumulative_3d, points3D))

# Extract the bounds between which the object's dimensions are needed
# It is necessary for this demo that the object's length and breadth is smaller than that of the chessboard
chessboard_points_cumulative_3d = np.delete(chessboard_points_cumulative_3d, 0, 1)
roi_2D = get_boundary_corners_2D(chessboard_points_cumulative_3d)

print("Calibration completed... \nPlace the box in the field of view of the devices...")

# Enable the emitter of the devices
device_manager.enable_emitter(True)

# Load the JSON settings file in order to enable High Accuracy preset for the realsense
device_manager.load_settings_json("./HighResHighAccuracyPreset.json")

# Get the extrinsics of the device to be used later
extrinsics_devices = device_manager.get_depth_to_color_extrinsics(frames)

# Get the calibration info as a dictionary to help with display of the measurements onto the color image instead of infra red image
calibration_info_devices = defaultdict(list)
for calibration_info in (transformation_devices, intrinsics_devices, extrinsics_devices):
    for key, value in calibration_info.items():
        calibration_info_devices[key].append(value)

def object_detection(color_image):
    image_expanded = np.expand_dims(color_image, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    vis_util.visualize_boxes_and_labels_on_image_array(
        color_image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return color_image

def volume_measurement(frames_devices):
    # Calculate the pointcloud using the depth frames from all the devices
    point_cloud = calculate_cumulative_pointcloud(frames_devices, calibration_info_devices, roi_2D)
    clusters = cluster_pointcloud(point_cloud)
    # Get the bounding box for the pointcloud in image coordinates of the color imager
    bounding_box_points_color_image, length, width, height = calculate_boundingbox_points(clusters, calibration_info_devices)
    
    # Create a blank image for visualizing the bounding box
    blank_image = np.zeros((resolution_height, resolution_width, 3), dtype=np.uint8)
    blank_image.fill(255)  # White background

    # Draw the bounding box on the blank image
    for point in bounding_box_points_color_image:
        cv2.circle(blank_image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

    cv2.putText(blank_image, f'Length: {length:.2f} mm', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(blank_image, f'Width: {width:.2f} mm', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(blank_image, f'Height: {height:.2f} mm', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return blank_image

try:
    while True:
        try:
            frames = pipeline.wait_for_frames()
        except RuntimeError:
            print("Frame didn't arrive, resetting pipeline...")
            start_pipeline()
            continue

        frames_devices = device_manager.poll_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Object detection
        color_image = np.asanyarray(color_frame.get_data())
        color_image_with_detections = object_detection(color_image.copy())

        # Volume measurement
        volume_image = volume_measurement(frames_devices)
        
        # Display results
        cv2.imshow('RealSense Object Detection', color_image_with_detections)
        cv2.imshow('Volume Measurement', volume_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    if pipeline_started:
        pipeline.stop()
    cv2.destroyAllWindows()
