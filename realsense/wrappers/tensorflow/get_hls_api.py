import tensorflow as tf

# 모델 파일 경로

model_path = 'D:\다운로드\GAY.h5'

# 모델 불러오기
model = tf.keras.models.load_model(model_path)
# 모델 요약 출력
print(model.summary())
