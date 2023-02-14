import tensorflow as tf # import 명령어

# tf v1처럼 사용하기
# Session()은 tf v2에서 지원하지 않는다.
# random_normal은 tf v2에서 random.normal로 바뀌었다.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 1. Build graph using TF operations

# H(x) = Wx + b

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# trainable Variable 선언
# tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# shape: The shape of the output tensor.
# mean: The mean of the normal distribution.
# stddev: The standard deviation of the normal distribution.
# dtype: The float type of the output: float16, bfloat16, float32, float64. Defaults to float32.
# seed: A Python integer. Used to create a random seed for the distribution.
# name: A name for the operation (optional).
# random.normal([shape 수], 이름). 
# 0~1 사이의 정규 확률분포값을 생성해주는 함수. 원하는 shape대로 만들어줌
# 정규분포란 가운데가 높고 양극단으로 갈수록 낮아져서 종 모양을 그리는 분포이다.
W = tf.Variable(tf.random.normal([1], name='weight'))
b = tf.Variable(tf.random.normal([1], name='blas'))

# Our hypothesis XW + b 정의
hypothesis = x_train * W + b

# cost(W,b) = 1/m * sigma^{m}_{i=1}(H(x^(i)) - y^(i))^2

# cost/Loss function
# cost 정의
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
# reduce_mean(array): array의 원소 평균을 내줌
# square(a - b) = (a - b)^2

# Minimize (GradientDescent)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
# train은 cost에 연결, cost는 hypothesis에 연결, W와 b 연결...


# 2&3. Run/update graph and get results

# Launch the graph in a session

sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):    # 2001번 반복
    sess.run(train) # 학습이 일어남
    if step % 20 == 0:  # 20번 마다 출력하라
        print(step, sess.run(cost), sess.run(W), sess.run(b))
        # 반복횟수, cost, W, b
        # 0~2000, 0에 수렴, [1에 수렴], [0에 수렴]