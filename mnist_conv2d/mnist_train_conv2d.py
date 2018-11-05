# _*_ coding:utf-8 _*_
"""定义神经网络的训练过程"""

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference_conv2d
import numpy as np

# 配置神经网络参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 保存模型的路径和文件名
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    """使用卷积神经网络时 对输入格式进行调整"""
    # x = tf.placeholder(tf.float32,
    #                    [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32,
                        [None, mnist_inference_conv2d.OUTPUT_NODE], name='y-input')
    regularization = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        mnist_inference_conv2d.IMAGE_SIZE,
        mnist_inference_conv2d.IMAGE_SIZE,
        mnist_inference_conv2d.NUM_CHANNELS,

    ], name='x-input')

    y = mnist_inference_conv2d.inference(input_tensor=x, train=True, regularizer=regularization)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作及训练过程
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,
                                                                   labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
                                                                           global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        # 进行训练，测试过程在另外一个程序中实现
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          mnist_inference_conv2d.IMAGE_SIZE,
                                          mnist_inference_conv2d.IMAGE_SIZE,
                                          mnist_inference_conv2d.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: reshaped_xs, y_: ys})

            # 每一千轮保存一次模型
            if i % 1000 == 0:
                print("After %d training step(s), loss on training "
                      "batch is %g." % (step, loss_value))

                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                    global_step=global_step
                )

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()