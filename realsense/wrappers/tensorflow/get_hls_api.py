import tensorflow as tf

def load_frozen_graph(pb_file_path):
    with tf.io.gfile.GFile(pb_file_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def print_graph_nodes(graph_def):
    for node in graph_def.node:
        print(node.name)

frozen_graph_path = "frozen_inference_graph.pb"
graph_def = load_frozen_graph(frozen_graph_path)
print_graph_nodes(graph_def)
