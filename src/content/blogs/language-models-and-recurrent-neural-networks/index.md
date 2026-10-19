---
title: "CS224N | 语言模型与循环神经网络"
publishDate: 2026-08-20
category: learning
tags: [NLP, CS224N]
language: zh
description: "CS224n lec4,A note on n-gram and neural language models, recurrent neural networks, bidirectional RNNs, GRUs, and LSTMs."
heroImage:
  src: /assets/img/posts/neural-network-architectures/transformer-card-cover.png
  alt: "文章封面"
  color: "#9a7093"
---

<aside class="download-callout" role="note">
  <p>You can download the <a href="/output/pdf/language-models-and-recurrent-neural-networks.tex">LaTeX</a> and <a href="/output/pdf/language-models-and-recurrent-neural-networks.pdf">PDF</a>.</p>
</aside>

> 人间 瞬间 天地之间 下次我 又是谁

## 语言模型
语言模型用于计算若干单词以某一特定顺序出现的概率，由 $m$ 个单词组成的序列 $\{w_1,...,w_m\}$ 的概率记为 $P(w_1,\dots,w_m)$，由于单词 $w_i$ 之 前出现的单词数量取决于它在输入文档中的位置，因此在实际计算中， $P(w_1, . . . , w_m)$ 通常不会条件于全部历史单词，而是近似地只条件于之前的 $n-1$ 个单词：

$$
P(w_1,\ldots,w_m)=\prod_{i=1}^{m}P(w_i\mid w_1,\ldots,w_{i-1})\approx\prod_{i=1}^{m}P(w_i\mid w_{i-n+1},\ldots,w_{i-1})
$$

### n-gram 语言模型
 为了计算上面的概率，可以把每个n-gram的出现次数与对应单词的频率作比较，这样就得到了n-gram语言模型，比如下面这个例子就是二元模型与三元模型之间的关系

$$
\begin{aligned}
p(w_2\mid w_1)&=\frac{\operatorname{count}(w_1,w_2)}{\operatorname{count}(w_1)}\\
p(w_3\mid w_1,w_2)&=\frac{\operatorname{count}(w_1,w_2,w_3)}{\operatorname{count}(w_1,w_2)}
\end{aligned}
$$

我们发现了可以用前 $n-1$ 个单词来预测第 $n$ 个单词，但是上下文窗口到底应该多长，在有些情况连续的前 $n$ 个单词不足以捕获真正的上下文。因此这就引出了n-gram模型两个主要的问题，**稀疏性**和**存储**

- n-gram模型的稀疏性问题：回顾上面提到的三元模型的例子，如果 $w_1,w_2,w_3$ 从未在语料库中共同出现，则 $w_3$ 的概率会被估计为0，一种解决方法是给词表中每个词的计数加一个很小的 $\delta$，这称为平滑。我们继续回顾这个等式，如 果 $w_1$ 和 $w_2$ 从未同时出现，那么 $w_3$ 的条件概率甚至无法计算。对此可以退而只条件于 $w_2$，这称为**回退**。随着 $n$ 增大，稀疏性会越来越严重，因此通常取 $n \le 5$
-  n-gram语言模型中的存储问题：我们需要保存语料库中出现过的所有n-gram的计数。因此当 $n$ 增大，或者语料库规模增大时，模型所需的存储空间也会随之增加。

### 基于窗口的神经语言模型
Bengio 等人在 *A Neural Probabilistic Language Model* 中提出使用神经网络处理语言模型中的“维数灾难”问题。模型同时学习单词的分布式表示，并利用前 $n$ 个单词的词向量预测下一个单词的概率分布。

![基于窗口的神经语言模型](/assets/img/posts/language-models-and-rnns/window-neural-language-model.png)

上图模型可写为

$$
\hat{y}=\operatorname{softmax}\left(W^{(2)}\tanh(W^{(1)}x+b^{(1)})+W^{(3)}x+b^{(3)}\right)
$$

其中，$W^{(1)}$ 作用于词向量，$W^{(2)}$ 作用于隐藏层，$W^{(3)}$ 直接作用于词向量。

![简化的神经语言模型](/assets/img/posts/language-models-and-rnns/simplified-neural-language-model.png)

上图简化后，将多个输入单词的词向量拼接为

$$
e=[e^{(1)};e^{(2)};e^{(3)};e^{(4)}]
$$

再经过隐藏层 $h=f(We+b_1)$，最后得到输出分布 $\hat{y}=\operatorname{softmax}(Uh+b_2)$

## 循环神经网络（RNN）
和传统语言模型只使用有限长度的历史窗口不同，RNN能够让模型条件于此前所有单词所携带的信息
下图展示了RNN的结构。

![循环神经网络结构](/assets/img/posts/language-models-and-rnns/rnn-structure.png)

每个竖直的矩形框表示时间步 $t$ 上的隐藏层，每个隐藏层包含若干神经元，每个神经元先对输入做线性矩阵运算，再通过非线性函数。在每个时间步，隐藏层有两个输入：上一时间步的隐藏状态 $h_{t-1}$ ，以及当前时间步的输入 $x_t$。前者乘以权重矩阵 $W^{(hh)}$，后者乘以权重矩阵 $W^{(hx)}$，共同产生隐藏特征 $h_t$ ，随后 $h_t$ 再乘以权重矩阵 $W^{(S)}$，并通过词表上的 softmax 得到下一个单词的预测分布 $\hat{y}_t$：

$$
\begin{aligned}
h_t&=\sigma\left(W^{(hh)}h_{t-1}+W^{(hx)}x_t\right)\\
\hat{y}_t&=\operatorname{softmax}\left(W^{(S)}h_t\right)
\end{aligned}
$$

每个时间步中都会重复使用同一组 $W^{(hh)}$ 和 $W^{(hx)}$ 。因此模型需要学习的参数数量更少，更重要的是，参数数量与输入序列的长度无关，避免了维数灾难

### RNN 的损失与困惑度
RNN 中常用的损失函数是前面讲义已经介绍过的交叉熵。时间步 $t$ 上，对整个词表求和可以得到

$$
\mathcal{J}^{(t)}(\theta)=-\sum_{j=1}^{\lvert V\rvert} y_{t,j}\log \hat{y}_{t,j}
$$

对于大小为 $T$ 的语料，整体交叉熵为

$$
J=\frac{1}{T}\sum_{t=1}^{T}\mathcal{J}^{(t)}(\theta)=-\frac{1}{T}\sum_{t=1}^{T}\sum_{j=1}^{\lvert V\rvert}y_{t,j}\log\hat{y}_{t,j}
$$

下式给出了**困惑度**。若交叉熵使用以 2 为底的对数，它等于 $2^J$；若使用自然对数，则等于 $e^J$：

$$
\mathrm{Perplexity}=2^J
$$

与真实结果相比，困惑度越低，意味着模型对下一个词的预测越有把握

### RNN 的局限性
RNN当前也不是全能的，它存在以下缺点

- 计算较慢，因为计算过程是顺序进行的，难以并行化；
- 实际中很难有效使用很久以前的信息，原因包括梯度消失和梯度爆炸

### 梯度消失与梯度爆炸
RNN会把权重矩阵从一个时间步反复传播到下一个时间步。RNN 的目标之一，就是让上下文信息能够跨越很远的时间距离传递，来自一段时间之前的信息在反向传播的过程中可能消失了，这就叫梯度消失，下面给出梯度消失的数学原因
在时间步 $t$ ，为了计算 RNN 关于参数 $W$ 的总误差梯度，需要把每个时间步的误差梯度累加

$$
\frac{\partial E}{\partial W}=\sum_{t=1}^{T}\frac{\partial E_t}{\partial W}
$$

对于单个时间步的梯度计算应用链式法则，可写成

$$
\frac{\partial E_t}{\partial W}=\sum_{k=1}^{t}\frac{\partial E_t}{\partial y_t}\frac{\partial y_t}{\partial h_t}\frac{\partial h_t}{\partial h_k}\frac{\partial h_k}{\partial W}
$$

其中 $\frac{\partial h_t}{\partial h_k}$ 表示第 $t$ 个隐藏状态对第 $k$ 个隐藏状态的梯度。根据链式法则，

$$
\frac{\partial h_t}{\partial h_k}=\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}=\prod_{j=k+1}^{t}W^T\operatorname{diag}\left[f'(h_{j-1})\right]
$$

因此总梯度为

$$
\frac{\partial E}{\partial W}=\sum_{t=1}^{T}\sum_{k=1}^{t}\frac{\partial E_t}{\partial y_t}\frac{\partial y_t}{\partial h_t}\left(\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}\right)\frac{\partial h_k}{\partial W}
$$

设 $\|W^T\|\leq\beta_W$，$\|\operatorname{diag}[f'(h_{j-1})]\|\leq\beta_h$，则

$$
\left\|\frac{\partial h_j}{\partial h_{j-1}}\right\|\leq\beta_W\beta_h,\qquad
\left\|\frac{\partial h_t}{\partial h_k}\right\|\leq(\beta_W\beta_h)^{t-k}
$$

因此：

- $\beta_W\beta_h<1$：梯度随时间指数衰减，产生**梯度消失**；
- $\beta_W\beta_h>1$：梯度可能指数增长，产生**梯度爆炸**。

### 梯度爆炸与梯度消失的解决方法
**梯度爆炸**

当梯度范数超过阈值时，将梯度缩放到固定范围：

$$
\hat g\leftarrow\frac{\partial E}{\partial W},\qquad
\|\hat g\|\geq\mathrm{threshold}\Rightarrow\hat g\leftarrow\frac{\mathrm{threshold}}{\|\hat g\|}\hat g
$$

我们把这个方法称为梯度裁剪，梯度裁剪保留梯度方向，只限制梯度大小，避免参数更新过大

**梯度消失**

常见方法：

- 将 $W^{(hh)}$ 初始化为单位矩阵，而不是随机初始化；
- 使用 ReLU 替代 sigmoid。ReLU 的导数为 $0$ 或 $1$，在导数为 $1$ 时，梯度不会随时间反向传播而不断衰减

### 深层双向 RNN
现在我们主要讨论了根据过去的单词预测下一个词的RNN。通过让另一个 RNN 反向读取语料，也可以利用未来单词的信息进行预测，普通 RNN 只利用过去的信息，双向 RNN 同时维护两个隐藏状态：

- $\overrightarrow{h_t}$：从左向右传播，编码过去信息；
- $\overleftarrow{h_t}$：从右向左传播，编码未来信息。

$$
\overrightarrow{h_t}=f\left(\overrightarrow{W}x_t+\overrightarrow{V}\overrightarrow{h}_{t-1}+\overrightarrow{b}\right),\qquad
\overleftarrow{h_t}=f\left(\overleftarrow{W}x_t+\overleftarrow{V}\overleftarrow{h}_{t+1}+\overleftarrow{b}\right)
$$

![双向循环神经网络](/assets/img/posts/language-models-and-rnns/bidirectional-rnn.png)

最终将两个方向的隐藏状态拼接后进行预测：

$$
\hat y_t=g\left(U[\overrightarrow{h_t};\overleftarrow{h_t}]+c\right)
$$

双向 RNN 能同时利用当前位置前后的上下文信息，但需要维护两套隐藏状态和参数。

**深层双向 RNN**

RNN 可以堆叠多层，较低层的输出作为上一层在同一时间步的输入。第 $i$ 层有

$$
\overrightarrow{h_t}^{(i)}=f\left(\overrightarrow{W}^{(i)}h_t^{(i-1)}+\overrightarrow{V}^{(i)}\overrightarrow{h}_{t-1}^{(i)}+\overrightarrow{b}^{(i)}\right),\qquad
\overleftarrow{h_t}^{(i)}=f\left(\overleftarrow{W}^{(i)}h_t^{(i-1)}+\overleftarrow{V}^{(i)}\overleftarrow{h}_{t+1}^{(i)}+\overleftarrow{b}^{(i)}\right)
$$

最后使用最高层两个方向的隐藏状态进行预测：

$$
\hat y_t=g\left(U[\overrightarrow{h_t}^{(L)};\overleftarrow{h_t}^{(L)}]+c\right)
$$

![深层双向循环神经网络](/assets/img/posts/language-models-and-rnns/deep-bidirectional-rnn.png)

## 门控循环单元（GRU）
除了上述扩展之外，人们发现使用更复杂的激活单元可以让RNN获得更好的表现。此前，我们一直通过仿射变换加逐点非线性函数，把隐藏状态 $h_{t-1}$ 转换成 $h_t$。这里则通过引入**门控激活函数**来改变RNN的结构
虽然理论上RNN可以捕获长期依赖，但是实践上很难训练到稳定实现这一点，GRU的设计让记忆更容易长期保持，因此更容易捕获长距离依赖。GRU根据 $h_{t-1}$ 和 $x_t$ 生成下一隐藏状态 $h_t$ 的过程为

1. **更新门**：决定过去的隐藏状态 $h_{t-1}$ 有多少直接保留到当前状态。

$$
z_t=\sigma(W^{(z)}x_t+U^{(z)}h_{t-1})
$$

当 $z_t\approx1$ 时，主要保留旧记忆；当 $z_t\approx0$ 时，主要使用新记忆。

2. **重置门**：决定生成新记忆时需要使用多少过去的信息。

$$
r_t=\sigma(W^{(r)}x_t+U^{(r)}h_{t-1})
$$

$r_t$ 越小，生成新记忆时越倾向于忽略过去的隐藏状态。

3. **新记忆生成**：将当前输入 $x_t$ 与经过重置门筛选后的过去信息结合，得到候选隐藏状态。

$$
\tilde h_t=\tanh(r_t\circ Uh_{t-1}+Wx_t)
$$

4. **隐藏状态**：根据更新门，在旧记忆 $h_{t-1}$ 和新记忆 $\tilde h_t$ 之间进行加权组合。

$$
h_t=(1-z_t)\circ\tilde h_t+z_t\circ h_{t-1}
$$

![GRU 的门控结构](/assets/img/posts/language-models-and-rnns/gru.png)

## 长短期记忆网络（LSTM）
LSTM通过**输入门、遗忘门和输出门**控制信息流，并额外维护记忆单元 $c_t$，用于保存长期信息。
可以将 LSTM 理解为以下几个阶段：

1. **新记忆生成**：根据当前输入 $x_t$ 和上一时刻隐藏状态 $h_{t-1}$ 生成候选记忆：

$$
\tilde c_t=\tanh\left(W^{(c)}x_t+U^{(c)}h_{t-1}\right)
$$

2. **输入门**：决定候选记忆 $\tilde c_t$ 中有多少信息写入当前记忆：

$$
i_t=\sigma\left(W^{(i)}x_t+U^{(i)}h_{t-1}\right)
$$

3. **遗忘门**：决定上一时刻记忆 $c_{t-1}$ 中有多少信息被保留：

$$
f_t=\sigma\left(W^{(f)}x_t+U^{(f)}h_{t-1}\right)
$$

4. **最终记忆生成**：结合保留的旧记忆和写入的新记忆，得到当前记忆 $c_t$：

$$
c_t=f_t\circ c_{t-1}+i_t\circ\tilde c_t
$$

5. **输出门与隐藏状态**：输出门决定记忆 $c_t$ 中有多少信息暴露到隐藏状态 $h_t$：

$$
o_t=\sigma\left(W^{(o)}x_t+U^{(o)}h_{t-1}\right),\qquad
h_t=o_t\circ\tanh(c_t)
$$

其中：

- $i_t$：控制**写入多少新信息**；
- $f_t$：控制**保留多少旧信息**；
- $o_t$：控制**输出多少当前记忆**；
- $c_t$：长期记忆；
- $h_t$：当前时刻对外输出的隐藏状态。

![LSTM 的门控结构](/assets/img/posts/language-models-and-rnns/lstm.png)
