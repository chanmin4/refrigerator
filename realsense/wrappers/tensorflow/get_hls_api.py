import tensorflow as tf

# Frozen Graph 파일 경로
frozen_graph_path = 'frozen_inference_graph.pb'

# 그래프 로드
with tf.io.gfile.GFile(frozen_graph_path, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# 노드 이름 출력
for node in graph_def.node:
    print(node.name)