import os
import sys
import time
import json
import cv2
sys.path.append(os.path.abspath("C:\\Users\\Jong Min Lee\\OneDrive\\Desktop\\github\\refrigerator\\realsense"))

import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
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
    # 리스트를 JSON 형식의 문자열로 변환
    items_json = json.dumps(detected_items)
    tag = EdgeTag(deviceId="volume_camera", tagName="object_recog", value=items_json)
    edgeData.tagList.append(tag)
    edgeAgent.sendData(edgeData)
    print(f"Sent detected objects to DataHub: {items_json}")

def run_detection():
    cap = cv2.VideoCapture(2)
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
    last_time_sent = time.time()  # 마지막으로 데이터를 전송한 시간 초기화

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

                image_np_expanded = np.expand_dims(frame, axis=0)
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})

                detected_classes = set()
                for i in range(int(output_dict['num_detections'][0])):
                    class_id = int(output_dict['detection_classes'][0][i])
                    score = float(output_dict['detection_scores'][0][i])
                    if score > 0.50:
                        class_name = category_index[class_id]['name']
                        detected_classes.add(class_name)

                current_time = time.time()
                if current_time - last_time_sent >= 5:  # 5초마다
                    if detected_classes:
                        send_data_to_datahub(list(detected_classes))
                        last_time_sent = current_time  # 마지막 전송 시간 업데이트

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
