# 神经网络数学基础
## 损失函数（loss function)
- 衡量预测值与期望值的误差
### 交叉熵
- 交叉熵刻画两个概率分布之间的距离
- 给定两个概率分布p和q,通过q来表示p的交叉熵为：(深度学习中P代表正确答案，q代表预测值）$$ H(p,q) = -\sum_{x} p(x) log q(x) $$
## 成本函数（cost function）
- 全部训练集损失函数的平均值
## 随机梯度下降（SGD:stochastic gradient descent)
- 给定一个初始点，并求出该点的梯度向量，以负梯度方向为搜索方向，以一定的步长进行搜索，从而确定下一个迭代点再计算该新梯度方向，如此重复直至Cost函数收敛。
- 令cost function 为 J(w,b), 欲令J(w,b) 取最小值
- 则需不断迭代：
$$ 
\begin{aligned}
w &= w - \alpha \frac{dJ(w,b)}{dw} \\\ 
b &= b - \alpha \frac{dJ(w,b)}{db}
\end{aligned}
$$
其中 α 为 learning rate

## 反向传播算法
- 反向传播从最终损失值开始，从最顶层反向作用到最底层，利用链式法则计算每个参数对损失值的贡献大小。
## 激活函数（activation function)
- 向网络中加入的非线性因素，加强网络的表示能力，解决下线性模型无法解决的问题
### 常用激活函数
- ReLU function: $$ f(x) = max(x,0) $$
- sigmoid function: $$ f(x) = \frac{1}{1+e^{-x}} $$
- tanh function: $$ f(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}} $$
