##################################################################################################
##       License: Apache 2.0. See LICENSE file in root directory.		                      ####
##################################################################################################
##                  Box Dimensioner with multiple cameras: Helper files 					  ####
##################################################################################################

import pyrealsense2 as rs
import numpy as np
import cv2
from realsense_device_manager import post_process_depth_frame
from helper_functions import convert_depth_frame_to_pointcloud, get_clipped_pointcloud
from sklearn.cluster import DBSCAN

def cluster_pointcloud(point_cloud, eps=0.05, min_samples=10):
	# 입력 데이터가 비어있는지 확인
	if point_cloud.size == 0 or point_cloud.shape[0] == 0:
		return []  # 빈 클러스터 리스트 반환

	db = DBSCAN(eps=eps, min_samples=min_samples).fit(point_cloud.T)
	labels = db.labels_
	unique_labels = set(labels)
	clusters = [point_cloud[:, labels == k] for k in unique_labels if k != -1]  # -1은 잡음을 의미합니다.
	return clusters


def calculate_boundingbox_points(clusters, calibration_info_devices, depth_threshold=0.01):
	# 최종 결과와 각 치수를 저장할 빈 리스트
	bounding_box_points_color_image = {}
	lengths, widths, heights = [], [], []

	for idx, cluster in enumerate(clusters): 
		if isinstance(cluster, np.ndarray) and cluster.shape[1] >= 3:
			print(f"Cluster {idx}: {cluster.shape[1]} points")  # 클러스터 크기 출력
			coord = cluster[:2, :].T.astype('float32')  # X, Y 좌표만 사용
			min_area_rect = cv2.minAreaRect(coord)
			box_points = cv2.boxPoints(min_area_rect)
			
			# 각 치수 계산
			width, height = min_area_rect[1]
			if width < height:
				width, height = height, width
			lengths.append(np.linalg.norm(box_points[0] - box_points[1]))
			widths.append(np.linalg.norm(box_points[1] - box_points[2]))
			heights.append(max(cluster[2,:]) - min(cluster[2,:]) + depth_threshold)
			
			height_array = np.array([[-height], [-height], [-height], [-height], [0], [0], [0], [0]])
			bounding_box_world_3d = np.column_stack((np.row_stack((box_points,box_points)), height_array))
			bounding_box_points_color_image={}
			# 이미지 좌표로 변환
			for device, calibration_info in calibration_info_devices.items():
				bounding_box_device_3d = calibration_info[0].inverse().apply_transformation(bounding_box_world_3d.transpose())
				color_pixel=[]
				bounding_box_device_3d = bounding_box_device_3d.transpose().tolist()
				for bounding_box_point in bounding_box_device_3d:
					bounding_box_color_image_point = rs.rs2_transform_point_to_point(calibration_info[2], bounding_box_point)
					color_pixel.append(rs.rs2_project_point_to_pixel(calibration_info[1][rs.stream.color], bounding_box_color_image_point))
				bounding_box_points_color_image[device] = np.row_stack( color_pixel )

	# 결과 처리: 아무런 클러스터도 처리되지 않았다면 기본값 반환
			return bounding_box_points_color_image, np.mean(lengths), np.mean(widths), np.mean(heights)
		else:
			return {}, 0, 0, 0  # 기본값 반환
def calculate_cumulative_pointcloud(frames_devices, calibration_info_devices, roi_2d, depth_threshold=0.01):
	"""
	Calculate the cumulative pointcloud from the multiple devices.
	Parameters:
	-----------
	frames_devices : dict
		The frames from the different devices.
		keys: Tuple of (serial, product-line)
		Serial number and product line of the device.
	values: [frame]
		frame: rs.frame()
		The frameset obtained over the active pipeline from the realsense device.

	calibration_info_devices : dict
		keys: str
		Serial number of the device.
		values: [transformation_devices, intrinsics_devices]
		transformation_devices: Transformation object
				The transformation object containing the transformation information between the device and the world coordinate systems.
		intrinsics_devices: rs.intrinscs
				The intrinsics of the depth_frame of the realsense device.

	roi_2d : array
		The region of interest given in the following order [minX, maxX, minY, maxY].

	depth_threshold : double
		The threshold for the depth value (meters) in world-coordinates beyond which the point cloud information will not be used.
		Following the right-hand coordinate system, if the object is placed on the chessboard plane, the height of the object will increase along the negative Z-axis.

	Return:
	----------
	point_cloud_cumulative : array
		The cumulative pointcloud from the multiple devices.
	"""
	# Use a threshold of 5 centimeters from the chessboard as the area where useful points are found
	point_cloud_cumulative = np.array([-1, -1, -1]).transpose()
	for (device_info, frame) in frames_devices.items():
		device = device_info[0]
		filtered_depth_frame = post_process_depth_frame(frame[rs.stream.depth], temporal_smooth_alpha=0.1, temporal_smooth_delta=80)
		point_cloud = convert_depth_frame_to_pointcloud(np.asarray(filtered_depth_frame.get_data()), calibration_info_devices[device][1][rs.stream.depth])
		point_cloud = np.asanyarray(point_cloud)

		point_cloud = calibration_info_devices[device][0].apply_transformation(point_cloud)
		point_cloud = get_clipped_pointcloud(point_cloud, roi_2d)
		point_cloud = point_cloud[:, point_cloud[2, :] < -depth_threshold]
		point_cloud_cumulative = np.column_stack((point_cloud_cumulative, point_cloud))
	
	point_cloud_cumulative = np.delete(point_cloud_cumulative, 0, 1)
	return point_cloud_cumulative



def visualise_measurements(frames_devices, bounding_box_points_devices, length, width, height):
	"""
 Calculate the cumulative pointcloud from the multiple devices
	
	Parameters:
	-----------
	frames_devices : dict
		The frames from the different devices
		keys: Tuple of (serial, product-line)
			Serial number and product line of the device
		values: [frame]
			frame: rs.frame()
				The frameset obtained over the active pipeline from the realsense device
				
	bounding_box_points_color_image : dict
		The bounding box corner points in the image coordinate system for the color imager
		keys: str
				Serial number of the device
			values: [points]
				points: list
					The (8x2) list of the upper corner points stacked above the lower corner points 
					
	length : double
		The length of the bounding box calculated in the world coordinates of the pointcloud
		
	width : double
		The width of the bounding box calculated in the world coordinates of the pointcloud
		
	height : double
		The height of the bounding box calculated in the world coordinates of the pointcloud
	"""
	for (device_info, frame) in frames_devices.items():
		device = device_info[0] #serial number
		color_image = np.asarray(frame[rs.stream.color].get_data())
		if device in bounding_box_points_devices:
			bounding_boxes = bounding_box_points_devices[device]
			for idx, bounding_box_points in enumerate(bounding_boxes):
				if (length != 0 and width !=0 and height != 0):
					bounding_box_points_device_upper = np.array(bounding_box_points_devices[device][0:4])
					bounding_box_points_device_lower = np.array(bounding_box_points_devices[device][4:8])
					box_info = "Length, Width, Height (mm): " + str(int(length*1000)) + ", " + str(int(width*1000)) + ", " + str(int(height*1000))
			# 상단 박스 그리기
					bounding_box_points_device_upper = tuple(map(tuple,bounding_box_points_device_upper.astype(int)))
					for i in range(len(bounding_box_points_device_upper)):	
						cv2.line(color_image, bounding_box_points_device_upper[i], bounding_box_points_device_upper[(i+1)%4], (0,255,0), 4)

					bounding_box_points_device_lower = tuple(map(tuple,bounding_box_points_device_lower.astype(int)))
					for i in range(len(bounding_box_points_device_upper)):	
						cv2.line(color_image, bounding_box_points_device_lower[i], bounding_box_points_device_lower[(i+1)%4], (0,255,0), 1)
			# 연결선 그리기
					for i in range(4):
						cv2.line(color_image, tuple(bounding_box_points_device_upper[i]), tuple(bounding_box_points_device_lower[i]), (0, 255, 0), 1)
					cv2.putText(color_image, box_info, (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0) )
			
		# Visualise the results
		cv2.imshow('Color image from RealSense Device Nr: ' + device, color_image)
		cv2.waitKey(1)
