import tensorflow as tf
import numpy as np
model_path = './saved_model'
detection_model = tf.saved_model.load(model_path)
print(detection_model.signatures)
detection_fn = detection_model.signatures['serving_default']

# 테스트 이미지 로드
input_tensor = tf.convert_to_tensor(np.zeros((1, 640, 480, 3)), dtype=tf.float32)
detections = detection_fn(input_1=input_tensor)

# 출력 키 확인
print(detections.keys())
