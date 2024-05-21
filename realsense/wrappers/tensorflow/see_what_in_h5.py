import tensorflow as tf

# TensorFlow-DirectML 설정
tf.compat.v1.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))