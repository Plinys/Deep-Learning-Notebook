"""从TFRecord文件读取数据"""
import tensorflow as tf

# 创建一个reader来读取TFRecored文件中的样例
reader = tf.TFRecordReader()
# 创建一个队列来维护输入文件列表
filename_queue = tf.train.string_input_producer(["./mnist_to_TFRecord.tfrecord"])

# 从文件中读取一个样例
_, serialized_example = reader.read(filename_queue)
# 解析读入的一个样例
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
    }
)

# tf.decode_raw 可以将字符串解析成图像对应的像素数组
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()
# 启动多线程处理数据
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 每次运行可以读取TFRecord文件中的样例
for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])
