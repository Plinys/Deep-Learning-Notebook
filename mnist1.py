# _*_ coding: utf-8 _*_
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 载入mnist数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
# 初始化w, b矩阵
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 前向传播得到预测值
y = tf.nn.softmax(tf.matmul(x, w) + b)

# 占位符用于输入正确值
y_ = tf.placeholder("float", [None, 10])

# 定义损失函数：交叉熵函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 采用梯度下降法以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化所有变量
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(50):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 计算精度值，将True和false转化为浮点值，求取平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()
