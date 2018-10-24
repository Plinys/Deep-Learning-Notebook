## tf.Session() 会话
- 会话拥有并管理Tensorflow程序运行时的所有资源
- 利用python中的上下文管理器来管理会话是一个很好的选择，不必调用Session.close()来关闭会话:with tf.Session() as sess
### Session.run() 和 Tensor.eval() 的区别
- 对于张量 t , t.eval() 和 tf.get_dafault_session().run(t) 一样，两种方式都会运行整个计算图
- 最大的不同在于可以利用Session.run() 获得多个不同张量的取值
- eval()只能用于tf.Tensor类对象，也就是有输出的Operation
## tf.random_normal([2,3], stddev=1, seed=1, dtype=float)
- 正太分布的随机矩阵，stddev:标准差，seed:随机种子，当被赋值时每次运行时随机数都一样
## tf.placeholder(dtype, shape, name)
- placeholder 相当于定义了一个位置，这个位置的数据在程序运行时再指定
- 也可以理解为形参，用于定义过程，在执行的时候再赋具体的值
## tf.clip_by_value(A, min, ) 
- 将张量中的每一个元素值压缩在min和max之间，小于min的值让它等于min,大于max的值让其等于max
## tf.greater(v1, v2)
- 将张量V1和V2中的每个元素进行对比，当V1中的元素大于V2时返回True否则返回Flase
## tf.select(condation, func1, func2)
- 第一个参数为选择条件，当条件为True时，返回第二个参数的值，当条件为Flase时返回第三个参数的值，选择和判断都是在元素级别进行的

