import os
import numpy as np
import tensorflow as tf
import cv2
import pyrealsense2 as rs
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

# Custom DepthwiseConv2D Layer 정의
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(**kwargs)

# 모델 로드
model = load_model('./model_trained.h5', custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

print("[INFO] Starting streaming...")
pipeline.start(config)
print("[INFO] Camera ready.")

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())

    # Resize image to the input size expected by the model
    input_size = (299, 299)  # 모델 입력 크기에 맞추기
    resized_image = cv2.resize(color_image, input_size)
    image_expanded = np.expand_dims(resized_image, axis=0)

    # Actual detection
    predictions = model.predict(image_expanded, verbose=0)


    # Assuming the model outputs class probabilities
    # Get the highest probability class
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Visualization of the results of a detection
    if confidence > 0.5:  # confidence threshold
        label = f"{confidence:.2f}"
        cv2.putText(color_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('RealSense Object Detection', color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
