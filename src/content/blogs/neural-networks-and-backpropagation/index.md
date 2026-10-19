---
title: "CS224N | 神经网络与反向传播"
publishDate: 2026-08-10
category: learning
tags: [NLP, CS224N]
language: zh
description: "CS224n lec3,neural networks, backpropagation, regularization, activation functions, and data preprocessing."
heroImage:
  src: /assets/img/posts/neural-network-architectures/transformer-card-cover.png
  alt: "文章封面"
  color: "#9a7093"
---

<aside class="download-callout" role="note">
  <p>You can download the <a href="/output/pdf/neural-networks-and-backpropagation.tex">LaTeX</a> and <a href="/output/pdf/neural-networks-and-backpropagation.pdf">PDF</a>.</p>
</aside>

> 寻找太阳的梦 自不量力说 自己也变成太阳的念头
# 神经网络与反向传播

## 神经网络

### 1.1 一个神经元

神经元接收 $n$ 个输入并产生一个输出。不同的神经元会产生不同的输出因为参数不同，比较常见的神经元是Sigmoid逻辑单元。该单元接收一个 $n$ 维输入向量 $x$，并产生一个输出值 $a$。这一神经元还具有一个 $n$ 维权重向量 $w$ 和一个标量偏置 $b$ 。输出为

$$
a=\frac{1}{1+\exp(-(w^\top x+b))}
$$

### 1.2 单层神经元

扩展到多个神经元的输入，设不同神经元的权重分别为$\{w^{(1)},w^{(2)},\dots,w^{(m)}\}$，偏置分别为$\{b_1,\dots,b_m\}$，对应的激活值为$\{a_1,\dots,a_m\}$

$$
a_1=\frac{1}{1+\exp(-(w^{(1)T}x+b_1))},
$$

$$
\vdots
$$

$$
a_m=\frac{1}{1+\exp(-(w^{(m)T}x+b_m))}
$$

为了使记号更简洁，并便于表示更复杂的网络，定义

$$
\sigma(z)=
\begin{bmatrix}
\dfrac{1}{1+\exp(-z_1)}\\
\vdots\\
\dfrac{1}{1+\exp(-z_m)}
\end{bmatrix},
\quad
b=
\begin{bmatrix}
b_1\\
\vdots\\
b_m
\end{bmatrix}
\in\mathbb{R}^m,
$$

$$
W=
\begin{bmatrix}
-\,w^{(1)T}-\\
\vdots\\
-\,w^{(m)T}-
\end{bmatrix}
\in\mathbb{R}^{m\times n}.
$$

因此输出可以转化为$z = Wx + b$，Sigmoid函数的激活值可以写成$\begin{bmatrix} a^{(1)}\\ \vdots\\ a^{(m)} \end{bmatrix} =\sigma(z)=\sigma(Wx+b)$

### 1.3 前馈计算

神经网络的前馈计算过程为$a=f(Wx+b)$，其中 $x$ 为输入，$W,b$ 为参数，$f$ 为激活函数，$a$ 为隐藏层激活值。再根据隐藏层得到输出分数$s=U^Ta=U^Tf(Wx+b)$，隐藏层的作用是通过非线性变换提取输入中的组合特征，使模型能够学习更复杂的关系。

### 1.4 最大间隔目标函数

与大多数机器学习模型一样，神经网络也需要一个优化目标，最大间隔目标的思想是确保真实标注数据点的得分高于错误标注数据点的得分，我们将正确标注点的记为 $s$，错误标注点的得分记为 $s_c$，其中

$$
s_c=U^Tf(Wx_c+b),\quad s=U^Tf(Wx+b)
$$

那么目标函数应最大化 $(s-s_c)$，等价于最小化 $(s_c-s)$，我们考虑 $s_c  > s$ 的情况下，目标就变成了 $minimize \space J = max(s_c - s,0)$，但是我们希望真实标注数据点至少比错误标注数据点高出一个正间隔 $\Delta$，所以目标变为

$$
\operatorname{minimize} J=\max(\Delta+s_c-s,0)
$$

### 1.5 反向传播

这一节讨论当代价 $J$ 为正的时候如何训练模型中的各个参数。若代价为0，无需更新任何参数。我们通常使用梯度下降来更新参数，因此需要得到更新公式中任意参数的梯度

$$
\theta^{(t+1)}=\theta^{(t)}-\alpha\nabla_{\theta^{(t)}}J
$$

我们可以利用微分链式法则计算前馈过程中任意参数对应损失梯度

![三层神经网络及其前馈连接](/assets/img/posts/neural-networks-and-backpropagation/network-architecture.png)

**记号说明**

| 记号             | 含义                                             |
| ---------------- | ------------------------------------------------ |
| $x_i$            | 神经网络的第 $i$ 个输入                          |
| $s$              | 神经网络的输出                                   |
| $z_j^{(k)}$      | 第 $k$ 层第 $j$ 个神经元接收的标量输入           |
| $a_j^{(k)}$      | 第 $k$ 层第 $j$ 个神经元产生的激活输出           |
| $\delta_j^{(k)}$ | 在 $z_j^{(k)}$ 处计算并向后传播的误差            |
| $W^{(k)}$        | 将第 $k$ 层的输出映射到第 $k+1$ 层输入的权重矩阵 |

其中，第 $1$ 层表示输入层，因此

$$
x_j=z_j^{(1)}=a_j^{(1)}.
$$

结合前面的记号，

$$
W^{(1)}=W,\quad W^{(2)}=U.
$$

下面开始推导，假设 $J = (1 + s_c - s)$ 为正，我们希望更新参数 $W_{14}^{(1)}$。它只影响隐藏层第一个神经元的输入 $z_1^{(2)}$，进而影响 $a_1^{(2)}$，因此向后传播的梯度只会受到那些由该参数贡献的量的影响

假设 $J=1+s_c-s>0$，由最大间隔损失可得 $\frac{\partial J}{\partial s}=-1$，$\frac{\partial J}{\partial s_c}=1$。 下面考虑第 $1$ 层权重 $W_{ij}^{(1)}$ 对输出 $s$ 的影响。根据链式法则，

$$
\begin{aligned} \frac{\partial s}{\partial W_{ij}^{(1)}} &=W_i^{(2)}\frac{\partial a_i^{(2)}}{\partial W_{ij}^{(1)}}\\ &=W_i^{(2)} \frac{\partial a_i^{(2)}}{\partial z_i^{(2)}} \frac{\partial z_i^{(2)}}{\partial W_{ij}^{(1)}}\\ &=W_i^{(2)}f'(z_i^{(2)}) \frac{\partial z_i^{(2)}}{\partial W_{ij}^{(1)}} \end{aligned}
$$

第 $2$ 层第 $i$ 个神经元的输入为 $z_i^{(2)}=b_i^{(1)}+\sum_k a_k^{(1)}W_{ik}^{(1)}$，因此 $\frac{\partial z_i^{(2)}}{\partial W_{ij}^{(1)}}=a_j^{(1)}$ 代入上式，

$$
\frac{\partial s}{\partial W_{ij}^{(1)}} =W_i^{(2)}f'(z_i^{(2)})a_j^{(1)}
$$

 定义第 $2$ 层第 $i$ 个神经元的反向传播误差为 $\delta_i^{(2)}=W_i^{(2)}f'(z_i^{(2)})$，于是

$$
\frac{\partial s}{\partial W_{ij}^{(1)}}=\delta_i^{(2)}a_j^{(1)}
$$

 其中，$\delta_i^{(2)}$ 是第 $2$ 层第 $i$ 个神经元的后向误差，$a_j^{(1)}$ 是前一层传入权重 $W_{ij}^{(1)}$ 的激活值。 因此可以概括为：权重梯度 = 后向误差 × 前向输入。

以参数 $W_{14}^{(1)}$ 为例，它只影响隐藏层第 $1$ 个神经元，因此误差沿对应路径反向传播：

1. 从输出 $a_1^{(3)}$ 开始，向后传播大小为 $1$ 的误差信号。

2. 误差经过 $z_1^{(3)}\to a_1^{(3)}$ 时，需要乘以局部梯度。这里局部梯度为 $1$，因此记

$$
\delta_1^{(3)}=1.
$$

3. 误差继续向前一层传播，通过权重 $W_1^{(2)}$ 分配给 $a_1^{(2)}$，因此此处误差为 $W_1^{(2)}\delta_1^{(3)}=W_1^{(2)}$。

4. 误差经过 $z_1^{(2)}\to a_1^{(2)}$ 时，再乘以激活函数的局部梯度 $f'(z_1^{(2)})$，于是

$$
\delta_1^{(2)}=f'(z_1^{(2)})W_1^{(2)}.
$$

5. 最后，$W_{14}^{(1)}$ 在前向传播中接收的输入为 $a_4^{(1)}$，因此它的梯度为

$$
\frac{\partial s}{\partial W_{14}^{(1)}}
   =\delta_1^{(2)}a_4^{(1)}
   =a_4^{(1)}f'(z_1^{(2)})W_1^{(2)}.
$$

**偏置更新**

偏置可以看作输入恒为 $1$ 的特殊权重，因此第 $k$ 层第 $i$ 个神经元的偏置梯度就是其反向传播误差：

$$
\frac{\partial J}{\partial b_i^{(k)}}=\delta_i^{(k)}
$$

 例如，对于偏置 $b_1^{(1)}$，其梯度为

$$
\frac{\partial J}{\partial b_1^{(1)}}=\delta_1^{(2)}=f'(z_1^{(2)})W_1^{(2)}
$$

**将 $\delta^{(k)}$传播到$\delta^{(k - 1)}$的一般步骤**

1. 第 $k$ 层第 $i$ 个神经元的误差为 $\delta_i^{(k)}$，它会沿权重 $W_{ij}^{(k-1)}$ 向前一层传播。

![误差信号沿加权连接反向传播](/assets/img/posts/neural-networks-and-backpropagation/backpropagation-connection.png)

2. 沿一条连接传播到第 $k-1$ 层第 $j$ 个神经元时，该路径贡献的误差为 $\delta_i^{(k)}W_{ij}^{(k-1)}$。

3. 由于第 $k-1$ 层第 $j$ 个神经元通常连接到下一层多个神经元，因此需要把所有路径的误差相加：

$$
\sum_i \delta_i^{(k)}W_{ij}^{(k-1)}.
$$

4. 得到总误差后，还需要乘以该神经元激活函数的局部梯度 $f'(z_j^{(k-1)})$。

5. 因此，第 $k-1$ 层第 $j$ 个神经元的误差为

$$
\delta_j^{(k-1)}
=f'(z_j^{(k-1)})
\sum_i \delta_i^{(k)}W_{ij}^{(k-1)}
$$

### 1.6 使用反向传播训练：向量化形式

到目前为止，我们讨论了如何计算模型中某个给定参数的梯度，下面将推广前面的做法，以便一次性更新权重向量

对于给定参数 $W_{ij}^{(k)}$，前面已经得到其误差梯度为 $\delta_i^{(k+1)} a_j^{(k)}$，因此整个矩阵 $W^{(k)}$ 的误差梯度为

$$
\nabla W^{(k)}=
\begin{bmatrix}
\delta_1^{(k+1)}a_1^{(k)} & \delta_1^{(k+1)}a_2^{(k)} & \cdots\\
\delta_2^{(k+1)}a_1^{(k)} & \delta_2^{(k+1)}a_2^{(k)} & \cdots\\
\vdots & \vdots & \ddots
\end{bmatrix}
=\delta^{(k+1)}a^{(k)T}
$$

因此，可以用“传播进入该矩阵的误差向量”和“由该矩阵向前传递 的激活向量”的外积，写出整个矩阵的梯度

$$
\delta^{(k)}=f'(z^{(k)})\circ\left(W^{(k)T}\delta^{(k+1)}\right)
$$

## 神经网络的实用技巧

### 2.1 正则化

和很多机器学习模型一样，神经网络很容易发生过拟合，一种常见的解决过拟合的方法是加入 $L2$ 正则化惩罚项，我们在

损失函数 $J$ 后附加一个额外项，这样总损失就变成了

$$
J_R=J+\lambda\sum_{i=1}^{L}\left\|W^{(i)}\right\|_F^2
$$

其中，$\left\|W^{(i)}\right\|_F$ 是矩阵 $W^{(i)}$ 的Frobenius范数，$\lambda$ 是一个超参数，用于控制正则项相对于原始损失函数的权重。由于目标是最小化 $J_R$，正则化本质上会在优化损失函数的同时，惩罚过大的权重。由于Frobenius范数具有二次性质，$L_2$ 正则化会有效降低模型的灵活性，从而缓解过拟合现象。

> 矩阵U的Frobenius范数定义为$\|U\|_F=\sqrt{\sum_i\sum_j U_{ij}^2}$

施加这种约束也可以解释为一种贝叶斯先验信念：最优权重应接近于0；具体多接近取决于 $\lambda$ 的取值，选择合适的 $\lambda$ 非常关键，必须通过超参数调优来确定。若 $\lambda$ 过大，大多数权重都会被压到非常接近 0，模型无法从训练数据中学到有意义的信息，通常会在训练集，验证集和测试集上都取得较差的准确率。若 $\lambda$ 过小，又会重新进入过拟合的状态。必须注意的是，偏置项不被正则化，也不对上述成本项做出贡献。因为偏置项 $b$ 只控制模型整体平移，不会明显增加模型复杂度；而权重 $w$ 决定模型对输入的敏感程度，更容易导致过拟合。

### 2.2 Dropout

**Dropout** 是一种强有力的正则化技术，其思想简单而有效：训练时，在每次前向/反向传播过程中，以概率 $(1-p)$ 随机丢弃一部分神经元，等价地，每个神经元以概率 $p$ 被保留。测试时，则使用完整网络计算预测结果。这样做通常会使网络从数据中学习到更有意义的信息，更不容易过拟合，并且往往能在当前任务上取得更高的总体性能，Dropout 实际上同时训练了指数数量的较小网络，并对它们的预测结果取平均。

实践中，引入 Dropout 的方式是：取每一层神经元的输出 $h$，以概率 $p$ 保留每个神经元，否则将其置为 0。在反向传播时，只让梯度通过前向传播中被保留的神经元。测试时，则使用网络中的所有神经元执行前向传播。

要让 Dropout 有效工作，还存在一个关键细节：测试时神经元输出的期望值，应当与训练时大致相同，否则输出的量级可能会发生巨大变化，网络的行为将不再明确。使用传统 Dropout 时，测试阶段将激活值乘以保留概率 $p$；使用倒置 Dropout 时，则在训练阶段把保留的激活值除以 $p$，测试时不再缩放。

### 2.3 神经元单元

我们现在讨论的神经网络都用 Sigmoid 神经元来引入非线性。然而，在许多应用中，采用其他激活函数的效果更好。下面列出一些常见的选择以及其函数与梯度定义；它们可以替代前面讨论的Sigmoid函数

**Sigmoid**：Sigmoid我们已经非常熟悉了，激活函数 $\sigma$ 定义为 $\sigma(z) = \frac{1}{1 + exp(-z)}$

其梯度为 $\sigma'(z) = \frac{\exp(-z)}{(1+\exp(-z))^2} = \sigma(z)(1-\sigma(z))$

![Sigmoid 激活函数](/assets/img/posts/neural-networks-and-backpropagation/sigmoid.png)

**Tanh**：Tanh函数是Sigmoid函数的一种替代选择，实践中常常收敛得更快。区别是Tanh的输出范围为-1到1，而Sigmoid为0到1

$$
tanh(z) = \frac{\exp(z)-\exp(-z)}{\exp(z)+\exp(-z)}=2\sigma(2z) - 1 \quad tanh(z) \in (-1,1)
$$

梯度为

$$
\tanh'(z)=1-\left(\frac{\exp(z)-\exp(-z)}{\exp(z)+\exp(-z)}\right)^2=1-\tanh^2(z)
$$

![Tanh 激活函数](/assets/img/posts/neural-networks-and-backpropagation/tanh.png)

**Hard Tanh：** 有时会使用 Hard Tanh 代替 Tanh，因为前者计算成本更低。不过，当 $\lvert z\rvert>1$ 时，它会饱和。Hard Tanh 的激活函数为

$$
\operatorname{hardtanh}(z)=
\begin{cases}
-1, & z<-1,\\
z, & -1\le z\le 1,\\
1, & z>1.
\end{cases}
$$

其导数也可写成分段函数：

$$
\operatorname{hardtanh}'(z)=
\begin{cases}
1, & -1\le z\le 1,\\
0, & \text{其他情况}.
\end{cases}
$$

![Hard Tanh 激活函数](/assets/img/posts/neural-networks-and-backpropagation/hard-tanh.png)

**Soft Sign：** Soft Sign 是另一种非线性函数，可以作为 Tanh 的替代，因为它不像硬截断函数那样容易饱和：

$$
\operatorname{softsign}(z)=\frac{z}{1+\lvert z\rvert}.
$$

其导数表示为

$$
\operatorname{softsign}'(z)=\frac{1}{(1+\lvert z\rvert)^2},
$$

![Softsign 激活函数](/assets/img/posts/neural-networks-and-backpropagation/softsign.png)

**ReLU：** ReLU是一种常见的激活函数。即使 $z$ 取较大的值，它也不会饱和，并且已经在计算机视觉应用中取得了很大成功：

$$
\operatorname{rect}(z)=\max(z,0).
$$

其导数是分段函数

$$
\operatorname{rect}'(z)=
\begin{cases}
1, & z>0,\\
0, & \text{其他情况}.
\end{cases}
$$

![ReLU 激活函数](/assets/img/posts/neural-networks-and-backpropagation/relu.png)

**Leaky ReLU：** 传统 ReLU 单元在 $z\le 0$ 时不会传播任何误差。Leaky ReLU 对此进行了修改，使得即使 $z$ 为负，仍允许少量误差向后传播：

$$
\operatorname{leaky}(z)=\max(z,kz),\qquad 0<k<1.
$$

因此，其导数可写为

$$
\operatorname{leaky}'(z)=
\begin{cases}
1, & z>0,\\
k, & \text{其他情况}.
\end{cases}
$$

![Leaky ReLU 激活函数](/assets/img/posts/neural-networks-and-backpropagation/leaky-relu.png)

### 2.4 数据预处理

和一般的机器学习模型一样，要确保模型在目标任务上取得合理性能，一个关键步骤是对数据进行基本预处理

**均值减法**

给定输入数据集合 $X$，通常会从 $X$ 中减去 $X$ 的平均特征向量，使数据以零为中心。一个重要细节是：实践中只使用训练集计算均值，然后从训练集、验证集和测试集中都减去该均值。

**归一化**
另一种常用技术——尽管可能不如均值减法常用——是把每个输入特征维度缩放到相近的数量级范围。这很有用，因为输入特征通常使用不同的“单位”进行度量，但我们往往希望在初始阶段把所有特征视为同等重要。实现方式很简单：用每个特征在训练集上计算得到的标准差，分别除以对应特征。

**白化**
白化不像“均值减法 + 归一化”那样常用。它本质上将数据转换为具有单位协方差矩阵的形式，即各特征彼此不相关，并且方差为 $1$。首先像往常一样对数据进行均值减法，得到 $X'$。然后对 $X'$ 作奇异值分解（SVD），得到矩阵 $U,S,V$。接着计算 $UX'$，将 $X'$ 投影到由 $U$ 的列定义的基中。最后，用 $S$ 中相应的奇异值除以结果的每个维度，从而适当地缩放数据；若某个奇异值为零，则可以改为除以一个很小的数。
