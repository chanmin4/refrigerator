import os
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

def create_tf_example(example):
    # 데이터 변환 코드 작성
    pass

def create_tfrecord(output_path, examples):
    writer = tf.io.TFRecordWriter(output_path)
    for example in examples:
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())
    writer.close()

# 데이터셋 디렉토리 경로
data_dir = "D:/다운로드/archive (3)"
train_output_path = os.path.join(data_dir, 'train.record')
val_output_path = os.path.join(data_dir, 'val.record')

# 데이터셋 로드 및 변환
train_examples = # 학습 데이터 로드
val_examples = # 검증 데이터 로드

create_tfrecord(train_output_path, train_examples)
create_tfrecord(val_output_path, val_examples)
