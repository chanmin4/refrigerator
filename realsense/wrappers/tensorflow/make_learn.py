import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# 경로 설정
model_path = './realsense/wrappers/tensorflow/frozen_inference_graph.pb'
label_map_path = './realsense/wrappers/tensorflow/mscoco_label_map.pbtxt'

image_path = './realsense/wrappers/tensorflow/green_apple.jpg'

# 허용된 클래스와 임계값 설정
ALLOWED_CLASSES = [44, 52, 53, 54, 55, 56, 57, 58, 59, 61]
LABEL_SCORE_THRESHOLDS = {
    'banana': 0.2,
    'apple': 0.2,
    'sandwich': 0.2,
    'orange': 0.2,
    'broccoli': 0.2,
    'carrot': 0.2,
    'hot dog': 0.2,
    'pizza': 0.2,
    'cake': 0.2,
    'bottle': 0.5
}

def load_model_and_labels(model_path, label_map_path):
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

def detect_objects(image_np, detection_graph, category_index):
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            tensor_dict = {
                'num_detections': detection_graph.get_tensor_by_name('num_detections:0'),
                'detection_boxes': detection_graph.get_tensor_by_name('detection_boxes:0'),
                'detection_scores': detection_graph.get_tensor_by_name('detection_scores:0'),
                'detection_classes': detection_graph.get_tensor_by_name('detection_classes:0')
            }

            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})

            num_detections = int(output_dict['num_detections'][0])
            detection_classes = []
            detection_boxes = []
            detection_scores = []
            for i in range(num_detections):
                score = float(np.squeeze(output_dict['detection_scores'])[i])
                class_id = int(np.squeeze(output_dict['detection_classes'])[i])

                if class_id not in ALLOWED_CLASSES:
                    continue

                if class_id not in category_index:
                    continue

                class_name = category_index[class_id]['name']
                
                if class_name in LABEL_SCORE_THRESHOLDS:
                    class_score_threshold = LABEL_SCORE_THRESHOLDS[class_name]
                else:
                    class_score_threshold = 0.5

                if score > class_score_threshold:
                    detection_classes.append(class_id)
                    detection_boxes.append(np.squeeze(output_dict['detection_boxes'])[i])
                    detection_scores.append(score)

            return {
                'num_detections': len(detection_classes),
                'detection_classes': np.array(detection_classes),
                'detection_boxes': np.array(detection_boxes),
                'detection_scores': np.array(detection_scores)
            }

def run_image_detection(image_path, model_path, label_map_path):
    detection_graph, category_index = load_model_and_labels(model_path, label_map_path)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_np = np.array(image_rgb)

    detected_objects = detect_objects(image_np, detection_graph, category_index)

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        detected_objects['detection_boxes'],
        detected_objects['detection_classes'].astype(np.int32),
        #detected_objects['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    cv2.imshow('Detection Results', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_image_detection(image_path, model_path, label_map_path)
