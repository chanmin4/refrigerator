# Import RealSense, OpenCV and NumPy
import pyrealsense2 as rs
import cv2
import numpy as np
import time
import base64
from collections import defaultdict
from realsense_device_manager import DeviceManager
from calibration_kabsch import PoseEstimation
from helper_functions import get_boundary_corners_2D
from measurement_task import cluster_and_bounding_box, calculate_boundingbox_points, calculate_cumulative_pointcloud, visualise_measurements
from sklearn.cluster import DBSCAN
from datahub_connector import edgeAgent  # Import the shared edgeAgent
from wisepaasdatahubedgesdk.Model.Edge import EdgeData, EdgeTag  # Ensure EdgeData and EdgeTag are imported

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

def run_demo():
    # Define some constants
    resolution_width = 848  # pixels
    resolution_height = 480  # pixels
    frame_rate = 30  # fps

    dispose_frames_for_stabilisation = 30  # frames

    chessboard_width = 6  # squares
    chessboard_height = 9  # squares
    square_size = 0.0253  # meters

    try:
        # Enable the streams from all the intel realsense devices
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
        rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

        # Use the device manager class to enable the devices and get the frames
        device_manager = DeviceManager(rs.context(), rs_config)
        device_manager.enable_all_devices()
        last_time_sent = time.time()
        # Allow some frames for the auto-exposure controller to stabilize
        for frame in range(dispose_frames_for_stabilisation):
            frames = device_manager.poll_frames()

        assert len(device_manager._available_devices) > 0
        """
        1: Calibration
        Calibrate all the available devices to the world coordinates.
        For this purpose, a chessboard printout for use with opencv based calibration process is needed.
        """
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
            
            visualize_calibration_status(frames, transformation_result_kabsch, intrinsics_devices, chessboard_params)

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
        # It is necessary for this demo that the object's length and breadth are smaller than that of the chessboard
        chessboard_points_cumulative_3d = np.delete(chessboard_points_cumulative_3d, 0, 1)
        roi_2D = get_boundary_corners_2D(chessboard_points_cumulative_3d)

        print("Calibration completed... \nPlace the box in the field of view of the devices...")

        """
        2: Measurement and display
        Measure the dimension of the object using depth maps from multiple RealSense devices
        The information from Phase 1 will be used here
        """

        # Enable the emitter of the devices
        device_manager.enable_emitter(True)

        # Load the JSON settings file in order to enable High Accuracy preset for the realsense
        device_manager.load_settings_json("./HighResHighAccuracyPreset.json")

        # Get the extrinsics of the device to be used later
        extrinsics_devices = device_manager.get_depth_to_color_extrinsics(frames)

        # Get the calibration info as a dictionary to help with display of the measurements onto the color image instead of infrared image
        calibration_info_devices = defaultdict(list)
        for calibration_info in (transformation_devices, intrinsics_devices, extrinsics_devices):
            for key, value in calibration_info.items():
                calibration_info_devices[key].append(value)

        # Continue acquisition until terminated with Ctrl+C by the user
        while True:
            # Get the frames from all the devices
            frames_devices = device_manager.poll_frames()
            current_time = time.time()
            if current_time - last_time_sent >= 5:
                for device_info, frame in frames_devices.items():
                    device = device_info[0]
                    color_image = np.asarray(frame[rs.stream.color].get_data())
                    #cv2.imshow('Color Image', color_image)
                    #cv2.waitKey(1)  # Add a small delay for the window to update
                    #변경해야함
                    #send_image_to_datahub(color_image, edgeAgent)
                last_time_sent = current_time

            # Calculate the pointcloud using the depth frames from all the devices
            point_cloud = calculate_cumulative_pointcloud(frames_devices, calibration_info_devices, roi_2D)
            clusters = cluster_and_bounding_box(point_cloud)
            # Get the bounding box for the pointcloud in image coordinates of the color imager
            bounding_box_points_color_image, length, width, height = calculate_boundingbox_points(clusters, calibration_info_devices)
            # Draw the bounding box points on the color image and visualise the results
            visualise_measurements(frames_devices, bounding_box_points_color_image, length, width, height)
            time.sleep(0.5)  # 2초 대기
    except KeyboardInterrupt:
        print("The program was interrupted by the user. Closing the program...")

    finally:
        device_manager.disable_streams()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()