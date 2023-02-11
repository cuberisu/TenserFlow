# Hello World!
import tensorflow as tf # import 명령어

# Session() 모듈 사용하기 (tf v1처럼 사용하기)
# Session()은 tf v2에서 지원하지 않는다.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Create a constant op
# This op is added as a node to the default graph
hello = tf.constant("Hello World!")

# seart a TF session
sess = tf.Session()

# run the op and get result
print(sess.run(hello))

# b'Hello World!'

# b'String' 'b' indicates Bytes literals.
# http://stackoverflow.com/questions/6269765