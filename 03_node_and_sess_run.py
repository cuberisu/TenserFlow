import tensorflow as tf # import 명령어

# tf v1처럼 사용하기
# Session()은 tf v2에서 지원하지 않는다.
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

# 1. Build graph(tensors) using TensorFlow operations
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)    # also tf.float32 implicitly
node3 = tf.add(node1, node2)

print("node1:", node1)
print("node2:", node2)
print("node3:", node3)

# node1: Tensor("Const_1:0", shape=(), dtype=float32) 
# (? 실제 출력값) node1: Tensor("Const:0", ...)
# node2: Tensor("Const_2:0", shape=(), dtype=float32)
# (? 실제 출력값) node2: Tensor("Const_1:0", ...)
# node3: Tensor("Add:0", shape=(), dtype=float32)

# 2. feed data and run graph (operation) 
# sess.run(op)

# 3. update variables in the graph (and return values)

sess = tf.Session()
print("sess.run(node1, node2):", sess.run([node1, node2]))
print("sess.run(node3):", sess.run(node3))

# sess.run(node1, node2): [3.0, 4.0]
# sess.run(node3): 7.0