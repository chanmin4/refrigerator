import os
import sys
# 경로 수정필요
sys.path.append(os.path.abspath("C:\\Users\\Jong Min Lee\\OneDrive\\Desktop\\github\\refrigerator\\realsense"))

import numpy as np
import tensorflow as tf
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from PIL import ImageFont

def run_detection():
    font = ImageFont.truetype("arial.ttf", 24)
    print(dir(font))  # 폰트 객체에서 사용할 수 있는 모든 메소드와 속성 나열

    # 웹캠 초기화
    cap = cv2.VideoCapture(1)  # 대부분의 경우 기본 웹캠은 인덱스 0을 사용합니다.
    if not cap.isOpened():
        print("Webcam not detected. Exiting...")
        return

    print("[INFO] Starting video stream from webcam...")
    print("[INFO] Camera ready.")

    # Model and label paths
    PATH_TO_CKPT = './frozen_inference_graph.pb'
    PATH_TO_LABELS = './mscoco_label_map.pbtxt'
    NUM_CLASSES = 90

    # Load the label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the TensorFlow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.compat.v1.import_graph_def(od_graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)

    # Detection
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert frame to numpy array
        color_image = np.asanyarray(frame)
        image_expanded = np.expand_dims(color_image, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # Visualization of the results of a detection
        vis_util.visualize_boxes_and_labels_on_image_array(
            color_image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        cv2.imshow('RealSense Object Detection', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    sess.close()

if __name__ == "__main__":
    run_detection()
