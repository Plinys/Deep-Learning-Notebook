### 通过最简单的逻辑回归的例子来推导反向传播具体过程
- 给定输入数据为x, 权重w, 偏执b, 线性变换输出为z, 预测值为a, 实际值为y, 损失函数为L(a,y)
$$
\begin{aligned}
& z = wx + b \\\\
& a = \sigma (z) = \frac{1}{1 + e^{-z}} \\\\
& L(a,y) = -ylog(a) + (1-y)log(1-a)
\end{aligned}
$$
- 想要使得预测值接近实际值，即损失函数取尽可能小，就需要不断调整权重w和偏置b来获得一个准确的预测值，这就是学习的过程
- 如何对w, b进行调整呢？我们可以对损失函数求w,b的偏导，从而来看其微小变化对损失函数的影响
- 接下来我们来看看反向传播的链式法则，看其如何求得dw
$$
\begin{aligned}
dw &= \frac{\partial L(a,y)}{\partial w} \\\\
& = \frac{\partial L(a,y)}{\partial a} \frac{\partial a}{\partial z} \frac{\partial z}{\partial w}
\end{aligned}
$$
其中
$$
\frac{\partial L(a,y)}{\partial a} = -\frac{y}{a} + \frac{1-y}{1-a}
$$
$$
\begin{aligned}
\frac{\partial a}{\partial z} &= ( \frac{1}{1+e^{-z}})^{'} \\\\
&= \frac{e^{-z}}{ (1 + e^{-z})^2 } \\\\
&= \frac{1+e^{-z}-1}{( 1 + e^{-z})^2 } \\\\
&= \frac{1}{1 + e^{-z}} - \frac{1}{ (1+e^{-z})^2 } \\\\
&= a(1-a)
\end{aligned}
$$
$$
\frac{\partial z}{\partial w} = x
$$
所以
$$
\begin{aligned}
dw& = \frac{\partial L(a,y)}{\partial w} \\\\
& = \frac{\partial L(a,y)}{\partial a} \frac{\partial a}{\partial z} \frac{\partial z}{\partial w} \\\\
& = (-\frac{y}{a} + \frac{1-y}{1-a}） a(1 - a) x \\\\
& = ( -y(1-a) + a(1-y) ) x \\\\
& = (-y + ya -ya + a) x \\\\
& = (a - y) x
\end{aligned}
$$
