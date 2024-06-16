import pyrealsense2 as rs
import numpy as np
import cv2
import time
from realsense_device_manager import post_process_depth_frame
from helper_functions import convert_depth_frame_to_pointcloud, get_clipped_pointcloud
from sklearn.cluster import DBSCAN
from datahub_connector import edgeAgent
from wisepaasdatahubedgesdk.Model.Edge import EdgeData, EdgeTag
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt  # Matplotlib 추가
time.sleep(5)  # 연결이 설정될 때까지 대기

last_time_sent = time.time()  # 마지막으로 데이터를 전송한 시간 초기화

def visualize_point_cloud(point_cloud, title="Point Cloud"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[0, :], point_cloud[1, :], point_cloud[2, :], s=1, c=point_cloud[2, :], cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

def send_data_to_datahub(total_volume_l):
    edgeData = EdgeData()
    deviceId = 'volume_camera'
    tagName = 'volume'
    tag = EdgeTag(deviceId, tagName, total_volume_l)
    edgeData.tagList.append(tag)
    edgeAgent.sendData(edgeData)
    print(f"Total volume {total_volume_l} L sent to DataHub")

def post_process_clusters(clusters, min_cluster_size=100):
    processed_clusters = []
    for cluster in clusters:
        if cluster.shape[1] >= min_cluster_size:
            processed_clusters.append(cluster)
    return processed_clusters

def statistical_outlier_removal(point_cloud, mean_k=20, std_dev_mul_thresh=1.0):
    if point_cloud.shape[1] < mean_k:
        return point_cloud

    kdtree = cKDTree(point_cloud.T)
    distances, _ = kdtree.query(point_cloud.T, k=mean_k)
    mean_distances = np.mean(distances, axis=1)
    std_distances = np.std(mean_distances)
    distance_threshold = np.mean(mean_distances) + std_dev_mul_thresh * std_distances

    filtered_points = point_cloud[:, mean_distances < distance_threshold]
    return filtered_points

def segment_objects_using_color(image, depth_image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([180, 255, 30])
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    mask_inv = cv2.bitwise_not(mask)
    segmented_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask_inv)
    return segmented_depth

def cluster_and_bounding_box(point_cloud):
    if point_cloud.size == 0:
        return []  # 포인트 클라우드가 비어 있으면 빈 리스트를 반환합니다.

    db = DBSCAN(eps=0.05, min_samples=10).fit(point_cloud.T)
    labels = db.labels_
    unique_labels = set(labels)
    clusters = [point_cloud[:, labels == k] for k in unique_labels if k != -1]
    clusters = post_process_clusters(clusters)
    return clusters  # clusters는 numpy 배열입니다.

def calculate_cumulative_pointcloud(frames_devices, calibration_info_devices, roi_2d, depth_threshold=0.02):
    point_cloud_cumulative = np.array([-1, -1, -1]).transpose()
    
    for (device_info, frame) in frames_devices.items():
        device = device_info[0]
        # filtered_depth_frame = post_process_depth_frame(frame[rs.stream.depth], temporal_smooth_alpha=5, temporal_smooth_delta=200)
        depth_frame = frame[rs.stream.depth]
        point_cloud = convert_depth_frame_to_pointcloud(np.asarray(depth_frame.get_data()), calibration_info_devices[device][1][rs.stream.depth])
        point_cloud = np.asanyarray(point_cloud)

        # 포인트 클라우드를 실제 좌표로 변환
        point_cloud = calibration_info_devices[device][0].apply_transformation(point_cloud)
        
        # 포인트 클라우드 시각화
        #visualize_point_cloud(point_cloud, title=f"Point Cloud from Device {device}")
        
        # ROI 내의 포인트만 유지
        point_cloud = get_clipped_pointcloud(point_cloud, roi_2d)
        # 포인트 클라우드 시각화
        #visualize_point_cloud(point_cloud, title=f"Point Cloud from Device {device}")
               
        # 특정 깊이 임계값을 적용하여 포인트 필터링 (depth 필터링)
        #z_mean = np.mean(point_cloud[2, :])
        point_cloud = point_cloud[:, point_cloud[2, :] < -depth_threshold]
        #최대 Z값을 기준으로 필터링
        #if point_cloud.size > 0:
            # 특정 깊이 임계값을 적용하여 포인트 필터링 (z축 필터링)
        #   max_z = np.max(point_cloud[2, :])
        #   point_cloud = point_cloud[:, point_cloud[2, :] < max_z + depth_threshold]
        
        # 포인트 클라우드 시각화
        #visualize_point_cloud(point_cloud, title=f"Point Cloud from Device {device}")
        
        point_cloud_cumulative = np.column_stack((point_cloud_cumulative, point_cloud))
    
    point_cloud_cumulative = np.delete(point_cloud_cumulative, 0, 1)
    return point_cloud_cumulative

def calculate_boundingbox_points(clusters, calibration_info_devices, depth_threshold=0.01):
    global last_time_sent
    bounding_box_points_color_image = {}
    lengths, widths, heights = [], [], []
    volumes = []

    for idx, cluster in enumerate(clusters):
        if isinstance(cluster, np.ndarray) and cluster.shape[1] >= 3:
            coord = cluster[:2, :].T.astype('float32')
            min_area_rect = cv2.minAreaRect(coord)
            box_points = cv2.boxPoints(min_area_rect)
            width, height = min_area_rect[1]
            if width < height:
                width, height = height, width
            length = np.linalg.norm(box_points[0] - box_points[1])
            widths.append(np.linalg.norm(box_points[1] - box_points[2]))
            z_values = cluster[2, :]
            max_height = max(z_values) - min(z_values)
            
            lengths.append(length)
            length_mm = length * 1_000
            width_mm = width * 1_000
            #보정작업 *10추가
            max_height_mm = max_height * 1_0000  # mm로 변환
            heights.append(max_height_mm)  # heights 리스트에 추가
            volume_mm3 = length_mm * width_mm * max_height_mm
            volumes.append(volume_mm3)
            height_array = np.array([[-max_height], [-max_height], [-max_height], [-max_height], [0], [0], [0], [0]])
            bounding_box_world_3d = np.column_stack((np.row_stack((box_points, box_points)), height_array))
            for device, calibration_info in calibration_info_devices.items():
                bounding_box_device_3d = calibration_info[0].inverse().apply_transformation(bounding_box_world_3d.transpose())
                color_pixel = []
                bounding_box_device_3d = bounding_box_device_3d.transpose().tolist()
                for bounding_box_point in bounding_box_device_3d:
                    bounding_box_color_image_point = rs.rs2_transform_point_to_point(calibration_info[2], bounding_box_point)
                    color_pixel.append(rs.rs2_project_point_to_pixel(calibration_info[1][rs.stream.color], bounding_box_color_image_point))
                if device not in bounding_box_points_color_image:
                    bounding_box_points_color_image[device] = []  # Initialize list for each device
                bounding_box_points_color_image[device].append(np.row_stack(color_pixel))
    if lengths and widths and heights:
        """
        total_volume_mm3 = sum(volumes)
        total_volume_l = total_volume_mm3 / 1_000_000
        current_time = time.time()
        if current_time - last_time_sent >= 5:
            send_data_to_datahub(total_volume_l)
            for idx, volume in enumerate(volumes):
                print(f"Cluster {idx}")
                print(f"Length = {lengths[idx] * 1_000:.2f} mm, Width = {widths[idx] * 1_000:.2f} mm, Height = {heights[idx]:.2f} mm, Volume = {volume:.2f} cubic millimeters")
            last_time_sent = current_time
        """
        return bounding_box_points_color_image, np.mean(lengths) * 1_000, np.mean(widths) * 1_000, np.mean(heights)
    else:
        return {}, 0, 0, 0
def visualise_measurements(frames_devices, bounding_box_points_devices, length, width, height):
    processed_images = {}
    for (device_info, frame) in frames_devices.items():
        
        device = device_info[0]
        color_image = np.asarray(frame[rs.stream.color].get_data()).copy()  # 프레임 복사
        if device in bounding_box_points_devices:
            bounding_boxes = bounding_box_points_devices[device]
            for idx, bounding_box_points in enumerate(bounding_boxes):
                if (length != 0 and width != 0 and height != 0):
                    bounding_box_points_device_upper = np.array(bounding_box_points_devices[device][idx][0:4])
                    bounding_box_points_device_lower = np.array(bounding_box_points_devices[device][idx][4:8])
                    box_info = "Length, Width, Height (mm): " + str(int(length)) + ", " + str(int(width)) + ", " + str(int(height))
                    bounding_box_points_device_upper = tuple(map(tuple, bounding_box_points_device_upper.astype(int)))
                    for i in range(len(bounding_box_points_device_upper)):
                        cv2.line(color_image, bounding_box_points_device_upper[i], bounding_box_points_device_upper[(i + 1) % 4], (0, 255, 0), 4)

                    bounding_box_points_device_lower = tuple(map(tuple, bounding_box_points_device_lower.astype(int)))
                    for i in range(len(bounding_box_points_device_upper)):
                        cv2.line(color_image, bounding_box_points_device_lower[i], bounding_box_points_device_lower[(i + 1) % 4], (0, 255, 0), 1)
                    cv2.line(color_image, bounding_box_points_device_upper[0], bounding_box_points_device_lower[0], (0, 255, 0), 1)
                    cv2.line(color_image, bounding_box_points_device_upper[1], bounding_box_points_device_lower[1], (0, 255, 0), 1)
                    cv2.line(color_image, bounding_box_points_device_upper[2], bounding_box_points_device_lower[2], (0, 255, 0), 1)
                    cv2.line(color_image, bounding_box_points_device_upper[3], bounding_box_points_device_lower[3], (0, 255, 0), 1)
                    cv2.putText(color_image, box_info, (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
        processed_images[device] = color_image  # 가공된 이미지를 저장
    
    return processed_images