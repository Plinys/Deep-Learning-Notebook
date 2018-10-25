### 通过最简单的逻辑回归的例子来推导反向传播具体过程
- 给定输入数据为x, 权重w, 偏执b, 线性变换输出为z, 预测值为a, 实际值为y, 损失函数为L(a,y)
$$
\begin{aligned}
& z = wx + b \\\\
& a = \sigma (z) = \frac{1}{1 + e^{-z}} \\\\
& L(a,y) = -ylog(a) + (1-y)log(1-a)
\end{aligned}
$$
