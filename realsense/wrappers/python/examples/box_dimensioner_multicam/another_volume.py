import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from realsense_device_manager import DeviceManager
def capture_frames(device_manager):
    frames = device_manager.poll_frames()
    depth_frame = frames[rs.stream.depth]
    depth_image = np.asanyarray(depth_frame.get_data())
    return depth_image

def post_process_depth_frame(depth_frame):
    # Apply filters to enhance the depth image (optional)
    spatial = rs.spatial_filter()
    depth_frame = spatial.process(depth_frame)
    return depth_frame

def detect_plane(depth_image, intrinsics):
    # Convert depth image to point cloud
    pcd = depth_image_to_pointcloud(depth_image, intrinsics)
    
    # Convert Open3D point cloud to numpy array
    pcd_np = np.asarray(pcd.points)

    # Use RANSAC to detect the largest plane
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    return plane_model, inliers, pcd

def depth_image_to_pointcloud(depth_image, intrinsics):
    height, width = depth_image.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_image / 1000.0  # Convert from mm to meters
    x = (x - intrinsics.ppx) * z / intrinsics.fx
    y = (y - intrinsics.ppy) * z / intrinsics.fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    
    # Filter out points with no depth
    points = points[z.flatten() > 0]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def separate_object_points(pcd, plane_model, distance_threshold=0.02):
    [a, b, c, d] = plane_model
    distances = np.abs(a * pcd.points[:, 0] + b * pcd.points[:, 1] + c * pcd.points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
    object_points = pcd.points[distances > distance_threshold]
    
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(object_points)
    return object_pcd

def calculate_volume(object_pcd):
    hull, _ = object_pcd.compute_convex_hull()
    volume = hull.get_volume()
    return volume

# Example usage
# Assuming `device_manager` and `intrinsics` are initialized
depth_image = capture_frames(device_manager)
depth_image = post_process_depth_frame(depth_image)
plane_model, inliers, pcd = detect_plane(depth_image, intrinsics)
object_pcd = separate_object_points(pcd, plane_model)

volume = calculate_volume(object_pcd)
print(f"Calculated volume: {volume:.2f} cubic meters")
