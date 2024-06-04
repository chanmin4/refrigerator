import os
import sys
path = "./realsense"
sys.path.append(os.path.abspath(path))
import time
import json
import cv2
import uuid
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from scipy.spatial import distance

from wisepaasdatahubedgesdk.EdgeAgent import EdgeAgent
from wisepaasdatahubedgesdk.Model.Edge import EdgeAgentOptions, DCCSOptions, EdgeData, EdgeTag, constant

# EdgeAgent 설정
edgeAgentOptions = EdgeAgentOptions(nodeId='3607ae3d-5e1e-4171-8706-b6a111fa05ac')
edgeAgentOptions.connectType = constant.ConnectType['DCCS']
dccsOptions = DCCSOptions(apiUrl='https://api-dccs-ensaas.sa.wise-paas.com/', credentialKey='8d47cc1fab2e0a5207ab7da336ae4atl')
edgeAgentOptions.DCCS = dccsOptions
edgeAgent = EdgeAgent(edgeAgentOptions)
edgeAgent.connect()

def send_data_to_datahub(detected_items):
    edgeData = EdgeData()
    items_json = json.dumps(detected_items)
    tag = EdgeTag(deviceId="volume_camera", tagName="object_recog", value=items_json)
    edgeData.tagList.append(tag)
    edgeAgent.sendData(edgeData)
    print(f"num of items:{len(detected_items)}")
    # 각 id별 클래스 이름을 출력
    for item in detected_items:
        
        print(f"Class Name:{item['class_name']}")
    
    print(f"Sent detected objects to DataHub: {items_json}")

def get_color_histogram(image, box):
    height, width = image.shape[:2]
    ymin, xmin, ymax, xmax = box
    x_min = int(xmin * width)
    y_min = int(ymin * height)
    x_max = int(xmax * width)
    y_max = int(ymax * height)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)
    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    hist = hist.astype(np.float32)  # 히스토그램을 float32로 변환
    return hist

def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    return cv2.compareHist(hist1, hist2, method)

def is_similar_object(hist1, hist2, threshold=0.7):
    similarity = compare_histograms(hist1, hist2)
    return similarity > threshold

def run_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not detected. Exiting...")
        return

    model_path = './realsense/frozen_inference_graph.pb'
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap('./realsense/mscoco_label_map.pbtxt')
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    last_time_sent = time.time()

    registered_objects = []

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_dict[key] = detection_graph.get_tensor_by_name(key + ':0')

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                original_height, original_width = frame.shape[:2]
                new_width = original_width * 2
                new_height = original_height * 2
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                image_np_expanded = np.expand_dims(frame, axis=0)
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})

                detected_classes = []
                for i in range(int(output_dict['num_detections'][0])):
                    class_id = int(output_dict['detection_classes'][0][i])
                    score = float(output_dict['detection_scores'][0][i])
                    if score > 0.50:
                        class_name = category_index[class_id]['name']
                        box = output_dict['detection_boxes'][0][i]
                        hist = get_color_histogram(frame, box)
                        if hist is None:
                            continue
                        centroid = [(box[1] + box[3]) / 2, (box[0] + box[2]) / 2]

                        similar_object_found = False
                        for obj in registered_objects:
                            if obj['class_name'] == class_name and is_similar_object(hist, obj['hist']):
                                if distance.euclidean(obj['centroid'], centroid) < 50:
                                    similar_object_found = True
                                    break

                        if similar_object_found:
                            detected_classes.append({'id': obj['id'], 'class_name': obj['class_name']})
                        else:
                            obj_id = str(uuid.uuid4())
                            registered_objects.append({'id': obj_id, 'class_name': class_name, 'hist': hist, 'centroid': centroid})
                            detected_classes.append({'id': obj_id, 'class_name': class_name})

                current_time = time.time()
                
                if current_time - last_time_sent >= 5:
                    if detected_classes:
                        send_data_to_datahub(detected_classes)
                        last_time_sent = current_time

                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(output_dict['detection_boxes']),
                    np.squeeze(output_dict['detection_classes']).astype(np.int32),
                    np.squeeze(output_dict['detection_scores']),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                cv2.imshow('RealSense Object Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()
