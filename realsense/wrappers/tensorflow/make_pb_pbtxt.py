#google colab전용파일

# Step 1: 라이브러리 설치 및 임포트
!pip install tensorflow
!pip install tensorflow_datasets

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.compat.v1 import Session, disable_eager_execution
from tensorflow.python.framework import graph_io
import numpy as np
import os
import time

# Step 2: GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU was found")

# Step 3: 데이터셋 로드 및 전처리
def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224)) / 255.0
    return image, label

def load_food101_dataset(batch_size=32):
    dataset, info = tfds.load('food101', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['validation']

    train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(1024).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, test_dataset, info

train_dataset, test_dataset, dataset_info = load_food101_dataset()
num_classes = dataset_info.features['label'].num_classes

# Step 4: 모델 구축
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

base_model.trainable = False

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Step 5: 모델 훈련
class TimeHistory(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        print(f"Starting epoch {epoch+1}")

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        print(f"Epoch {epoch+1} took {elapsed_time:.2f} seconds")
        if epoch == 0:
            self.first_epoch_time = elapsed_time

time_callback = TimeHistory()

epochs = 10
history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, callbacks=[time_callback])

# Step 6: 모델 저장 (frozen graph.pb 형식)
disable_eager_execution()
sess = Session()
tf.compat.v1.keras.backend.set_session(sess)
frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
    sess, 
    sess.graph.as_graph_def(), 
    [model.output.op.name.split(':')[0]]
)
graph_io.write_graph(frozen_graph, '.', 'frozen_inference_graph.pb', as_text=False)
print("Frozen graph saved at frozen_inference_graph.pb")

# Step 7: 라벨링 정보 저장 (pbtxt 형식)
label_map_path = 'mscoco_label_map.pbtxt'
labels = dataset_info.features['label'].names

def create_label_map(labels, output_path):
    with open(output_path, 'w') as f:
        for i, label in enumerate(labels):
            f.write('item {\n')
            f.write(f'  id: {i+1}\n')  # Label map IDs should start from 1
            f.write(f'  name: \'{label}\'\n')
            f.write('}\n')

create_label_map(labels, label_map_path)
print(f"Label map saved at {label_map_path}")

# Step 8: 모델과 라벨 파일을 Google Drive에 저장
from google.colab import drive
drive.mount('/content/drive')
!cp frozen_inference_graph.pb /content/drive/MyDrive/frozen_inference_graph.pb
!cp mscoco_label_map.pbtxt /content/drive/MyDrive/mscoco_label_map.pbtxt
