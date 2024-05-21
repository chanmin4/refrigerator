import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.compat.v1 import Session, disable_eager_execution
from tensorflow.python.framework import graph_io
import time

# 데이터 로드 및 전처리 함수
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU was found")

def get_file_paths_and_labels(json_file, image_dir, limit_per_class):
    with open(json_file, 'r') as file:
        data = json.load(file)

    file_paths = []
    labels = []
    label_map = {}
    
    for idx, (class_name, file_list) in enumerate(data.items()):
        label_map[class_name] = idx
        class_files = [os.path.join(image_dir, class_name, f"{fname}.jpg") for fname in file_list]
        
        if len(class_files) > limit_per_class:
            class_files = class_files[:limit_per_class]
        
        # 존재하는 파일만 추가
        valid_class_files = [f for f in class_files if os.path.exists(f)]
        file_paths.extend(valid_class_files)
        labels.extend([idx] * len(valid_class_files))
    
    return file_paths, labels, label_map

def preprocess_image(file_path, label):
    image = load_img(file_path.numpy().decode(), target_size=(224, 224))
    image = img_to_array(image) / 255.0
    return image, label

def tf_preprocess_image(file_path, label):
    image, label = tf.py_function(preprocess_image, [file_path, label], [tf.float32, tf.int32])
    image.set_shape((224, 224, 3))
    label.set_shape([])
    return image, label

def create_dataset(file_paths, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(tf_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(file_paths)).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# 메인 실행 부분
if __name__ == "__main__":
    image_dir = './food-101/images'
    train_json = './food-101/meta/train.json'
    test_json = './food-101/meta/test.json'
    
    train_file_paths, train_labels, train_label_map = get_file_paths_and_labels(train_json, image_dir, limit_per_class=200)
    test_file_paths, test_labels, test_label_map = get_file_paths_and_labels(test_json, image_dir, limit_per_class=200)

    train_dataset = create_dataset(train_file_paths, train_labels)
    test_dataset = create_dataset(test_file_paths, test_labels)

    # 모델 구축
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(len(train_label_map), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # 모델 컴파일
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 첫 번째 에포크 시간 측정
    class TimeHistory(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.start_time = time.time()
            print(f"Starting epoch {epoch+1}")

        def on_epoch_end(self, epoch, logs=None):
            elapsed_time = time.time() - self.start_time
            print(f"Epoch {epoch+1} took {elapsed_time:.2f} seconds")
            if epoch == 0:  # 첫 번째 에포크 시간 기록
                self.first_epoch_time = elapsed_time

    time_callback = TimeHistory()

    # 모델 훈련
    epochs = 10
    model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, verbose=1, callbacks=[time_callback])

    # 첫 번째 에포크 시간과 전체 예상 시간 출력
    first_epoch_time = getattr(time_callback, 'first_epoch_time', None)
    if first_epoch_time is not None:
        total_estimated_time = first_epoch_time * epochs
        print(f"Estimated total training time: {total_estimated_time / 60:.2f} minutes")
    else:
        print("First epoch time was not recorded.")

    # 모델 저장
    model.save('saved_model/my_model')

    # frozen inference graph 생성
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

    # label_map.pbtxt 생성 함수
    def create_label_map(labels_txt_path, output_path):
        with open(labels_txt_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        with open(output_path, 'w') as f:
            for i, label in enumerate(labels):
                f.write('item {\n')
                f.write('  id: {}\n'.format(i + 1))  # Label map IDs should start from 1
                f.write('  name: \'{}\'\n'.format(label))
                f.write('}\n')

    create_label_map('path/to/labels.txt', 'mscoco_label_map.pbtxt')
    print("Label map saved at mscoco_label_map.pbtxt")
