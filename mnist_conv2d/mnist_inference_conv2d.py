# _*_ coding:utf-8 _*_
"""定义卷积神经网络的参数及前向传播过程"""
import tensorflow as tf

# 配置神经网络的参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的深度和尺寸
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
# 全连接层的节点个数
FC_SIZE = 512

# 定义卷积神经网络的前向传播过程，添加新参数train,用于区分训练过程和测试过程
# 只有训练过程才会使用dropout 方法
def inference(input_tensor, train, regularizer):
    # 实现第一层卷积层的前向传播
    with tf.variable_scope('layer1_conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv1_biases = tf.get_variable(
            "bias", [CONV1_DEEP],
            initializer=tf.constant_initializer(0.0)
        )

        conv1 = tf.nn.conv2d(
            input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME'
        )
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 实现第二层池化层的前向传播
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(
            relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
        )

    # 实现第三层卷积层的前向传播
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_biases = tf.get_variable(
            "bias", [CONV2_DEEP],
            initializer=tf.constant_initializer(0.0)
        )
        conv2 = tf.nn.conv2d(
            pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME'
        )
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 实现第四层 池化层 前向传播
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(
            relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
        )

    # 全连接层的输入需要向量格式，因此需要将池化层的输出转换为向量格式
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 第五层 全连接层 同时引入dropout方法避免过拟合
    with tf.variable_scope('layer5-fc1'):
        fcl_weights = tf.get_variable(
            "weights", [nodes, FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fcl_weights))
        fc1_biases = tf.get_variable(
            "bias", [FC_SIZE], initializer=tf.constant_initializer(0.1)
        )
        fc1 = tf.nn.relu(tf.matmul(reshaped, fcl_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    # 第六层 全连接层
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable(
            "weight", [FC_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer != None:
            tf.add_to_collection('looses', regularizer(fc2_weights))

        fc2_biases = tf.get_variable(
            "bias", [NUM_LABELS],
            initializer=tf.constant_initializer(0.1)
        )
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

        return logit