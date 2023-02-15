import tensorflow as tf # import 명령어

# tf v1처럼 사용하기
# Session()은 tf v2에서 지원하지 않는다.
# placeholder는 tf v2에서 지원하지 않는다.
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

# 1. Build graph using TensorFlow operations
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

sess = tf.Session()

# 2. feed data and run graph (operation)
# sess.run(op, feed_dict={x: x_data})

# 3. update variables in the graph (and return values)
print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))
# 7.5
# [3. 7.]