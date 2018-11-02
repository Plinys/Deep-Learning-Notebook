import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8        # 基础学习率
LEARNING_RATE_DACAY = 0.99      # 学习率的衰减率
REGULARIZATION_RATE = 0.0001    # 正则化项在损失函数中的系数
TRAINING_STEPS = 30000          # 训练轮数
MOVING_AVERAGE_DACAY = 0.99     # 滑动平均衰减率

# 前向传播过程，一个使用RELU激活函数的三层全连接神经网络
# 可以选择是否使用滑动平均模型
def inference(input_tensor, avg_class, weights1, biases1,
              weights2, biases2):
    # 没有使用滑动平均模型
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) +
                      avg_class.average(biases1)
        )
        return tf.matmul(layer1, avg_class.average(weights2)) + \
               avg_class.average(biases2)

# 训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1)
    )
    biases1 = tf.Variable(
        tf.constant(0.1, shape=[LAYER1_NODE])
    )
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1)
    )
    biases2 = tf.Variable(
        tf.constant(0.1, shape=[OUTPUT_NODE])
    )

    # 未使用滑动平均值模型训练
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的变量，并指定为不可训练参数
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DACAY, global_step
    )
    # 在所有变量上使用滑动平均
    # tf.trainable_veriables 返回图上集合GraphKeys.TRAINABLE_VARIABLES的元素
    # 这个集合的元素是所有没有指定trainable=False 的参数
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 使用滑动平均模型的前向传播过程
    average_y = inference(
        x, variable_averages, weights1, biases1, weights2, biases2
    )

    # 损失函数：交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 损失函数添加L2正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization

    # 设置指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DACAY
    )

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 训练模型时，每过一遍数据需要通过反向传播来更新神经网络中的参数，
    # 又需要更新每一个参数的滑动平均值，为了一次完成多个操作，
    # tensorflow提供了tf.control_dependencies()和tf.group()两种机制
    # train_op = tf.group(train_step, variables_averages_op)是等价的
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            # 每一千轮输出一次在验证集上的测试结果
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d trainning step(s), validation accuracy"
                      "using average model is %g" % (i, validate_acc))
            # 产生这一轮使用的一个batch的训练数据，并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d trainning steps(s), test accuracy using averge"
              "model is %g" % (TRAINING_STEPS, test_acc))

# 主程序入口
def main(argv=None):
    # 载入数据集
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train(mnist)

# tensorflow 提供了一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()
