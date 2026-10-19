---
title: "CS224N | Transformer"
publishDate: 2026-08-21
category: learning
tags: [NLP, CS224N, Transformer]
language: zh
description: "CS224n lec5,A note on self-attention, positional representations, Transformer components, and encoder-decoder architectures."
heroImage:
  src: /assets/img/posts/neural-network-architectures/transformer-card-cover.png
  alt: "文章封面"
  color: "#9a7093"
---

<aside class="download-callout" role="note">
  <p>You can download the <a href="/output/pdf/neural-network-architectures-and-properties.tex">LaTeX</a> and <a href="/output/pdf/neural-network-architectures-and-properties.pdf">PDF</a>.</p>
</aside>

> 天雨粟 鬼夜哭 思念漫太古

## 基础

设 $w_{1:n}$ 表示长度为 $n$ 的词序列，其中 $w_i\in V$，$V$ 为有限词表。也可以将整个序列表示为 one-hot 向量组成的矩阵 $w_{1:n}\in\mathbb R^{n\times\lvert V\rvert}$。

语言模型根据前面的词预测当前词：

$$
w_t\sim\operatorname{softmax}\left(f(w_{1:t-1})\right)
$$

其中 $f(w_{1:t-1})\in\mathbb R^{\lvert V\rvert}$，softmax 将模型输出转换为词表上的概率分布。

对于 $A\in\mathbb R^{\ell\times d}$，softmax 通常沿最后一个维度计算：

$$
\operatorname{softmax}(A)_{i,j}=
\frac{\exp A_{i,j}}{\sum_{j'=1}^{d}\exp A_{i,j'}}
$$

对于更高维张量同理，例如 $B\in\mathbb R^{m\times\ell\times d}$，仍然沿最后一个维度计算 softmax。
**词嵌入**

设嵌入矩阵 $E\in\mathbb R^{d\times\lvert V\rvert}$，它将 one-hot 向量 $w$ 映射为 $d$ 维词向量 $x=Ew\in\mathbb R^d$。

对于整个序列，有 $x_{1:n}=w_{1:n}E^T\in\mathbb R^{n\times d}$。

**上下文化表示**

词嵌入 $Ew_i$ 是**非上下文化表示**，只由词 $w_i$ 本身决定，因此同一个词在不同上下文中的表示相同。

上下文化表示 $h_i$ 则同时取决于词本身和上下文，可写为 $h_i=f(x_{1:n})$；在语言模型中通常只使用前缀信息，即 $h_i=f(x_{1:i})$。

- **非上下文化表示**：只取决于词本身；
- **上下文化表示**：取决于词本身以及所在的上下文。

## 一种最简自注意力架构
广义而言，注意力是一种方法：给定一个query，通过挑选与查询最相似的键所对应的值，在键值 存储中进行软查找。这里所说的“挑选”和“最相似”，是指对所有值求平均，并为那些对应键更像查询的值赋予更高权重。在自注意力中，我们使用同一组元素来帮助定义查询、 键和值
### 键—查询—值自注意力机制
我们这里讨论的是一种最流行的自注意力机制，键-查询-值自注意力机制
对于序列 $x_{1:n}$ 中的词元 $x_i$，分别通过三个权重矩阵 $Q,K,V\in\mathbb R^{d\times d}$ 得到查询、键和值：

$$
q_i=Qx_i, k_j=Kx_j, v_j=Vx_j
$$

其中：

- $q_i$（Query）：表示当前词元想要寻找什么信息；
- $k_j$（Key）：表示词元 $x_j$ 能提供什么信息；
- $v_j$（Value）：表示词元 $x_j$ 实际提供的信息。

首先用 $q_i$ 和 $k_j$ 的点积衡量词元 $x_i$ 与 $x_j$ 的相关程度，再通过 softmax 得到注意力权重：

$$
\alpha_{ij}=
\frac{\exp(q_i^Tk_j)}
{\sum_{j'=1}^{n}\exp(q_i^Tk_{j'})}
$$

其中 $\alpha_{ij}$ 表示在计算 $x_i$ 的表示时，应该关注 $x_j$ 的程度。

最后对所有 Value 加权求和，得到 $x_i$ 的上下文化表示：

$$
h_i=\sum_{j=1}^{n}\alpha_{ij}v_j
$$

因此，Self-Attention 的核心过程可以概括为：
计算 Q、K、V → Query 与 Key 计算相关性 → softmax 得到权重 → 对 Value 加权求和。

![键—查询—值自注意力机制](/assets/img/posts/neural-network-architectures/self-attention.png)

### 位置表示
Self-Attention 本身没有内置的序列顺序概念。若只改变输入词元的排列，注意力计算中的相似度和加权求和也只是相应地重新排列，因此模型无法仅靠 Self-Attention 区分词元所在的位置。

也就是说，普通词嵌入 $x_i=Ew_i$ 只取决于词元本身，而不依赖它在序列中的位置。

**位置嵌入**

一种常见方法是为每个位置学习一个位置向量。设位置嵌入矩阵 $P\in\mathbb R^{N\times d}$，其中 $N$ 是模型支持的最大序列长度。

对于位置 $i$ 的词元，将位置向量 $P_i$ 加到词嵌入 $x_i$ 上：

$$
\tilde{x}_i=x_i+P_i
$$

随后使用 $\tilde{x}_i$ 进行 Self-Attention。这样同一个词出现在不同位置时会得到不同的输入表示，使模型能够感知序列顺序。

**在 Attention 中加入位置偏置**

除了修改输入表示，也可以直接修改注意力分数，使注意力本身具有位置概念。

一种方法是在原始注意力分数中加入与相对距离有关的偏置：

$$
\alpha_i=\operatorname{softmax}\left(K^Tq_i+b_i\right)
$$

其中 $b_i$ 根据词元之间的相对位置确定。通常距离当前位置越近的词元偏置越大，从而使模型更倾向于关注附近的信息。

### 逐元素非线性

如果只不断堆叠 Self-Attention 层，虽然每一层都会重新计算注意力权重，但从 Value 的聚合形式来看，多层结果仍可以整理成对输入表示的加权线性组合，因此仅靠堆叠注意力层不足以提供深度网络中丰富的非线性变换。

以两层 Self-Attention 为例，第二层输出可以写成

$$
o_i=\sum_{j=1}^{n}\alpha_{ij}V^{(2)}\left(\sum_{k=1}^{n}\alpha_{jk}V^{(1)}x_k\right)
=\sum_{k=1}^{n}\alpha_{ik}^{*}V^{*}x_k
$$

其中 $V^{*}=V^{(2)}V^{(1)}$，$\alpha_{ik}^{*}$ 表示两层注意力权重组合后的结果。可以看到，其形式仍与单层 Self-Attention 类似：对输入经过线性变换后再进行加权求和。

因此，在 Self-Attention 之后通常加入**逐元素前馈网络（Feed-Forward Network, FFN）**，为模型引入非线性：

$$
h_{\mathrm{FF}}=W_2\operatorname{ReLU}(W_1h_{\mathrm{self-attention}}+b_1)+b_2
$$

FFN 会对序列中的每个词元独立地执行相同的变换，即不同位置共享参数，但计算互不影响。

通常 FFN 先把隐藏维度 $d$ 扩展到更大的中间维度，再投影回 $d$ 维。例如 $W_1\in\mathbb R^{5d\times d}$，$W_2\in\mathbb R^{d\times5d}$。

### 因果掩码

在自回归语言模型中，模型根据当前位置之前的所有词预测下一个词：

$$
w_t\sim\operatorname{softmax}\left(f(w_{1:t-1})\right)
$$

其中 $f$ 根据前缀 $w_{1:t-1}$ 计算下一个词的概率分布。关键要求是：预测位置 $t$ 时不能使用未来位置的信息，否则模型在训练时就会提前看到答案。

RNN 天然满足这一要求，因为隐藏状态按时间顺序递推：

$$
h_{t-1}=\sigma(Wh_{t-2}+Ux_{t-1}),\
w_t\sim\operatorname{softmax}(h_{t-1}E)
$$

因此 $h_{t-1}$ 只包含 $1,\ldots,t-1$ 时刻的信息，不会看到未来。

而 Self-Attention 默认允许每个位置关注整个序列，因此需要加入**未来掩码（causal mask）**，限制位置 $i$ 只能关注满足 $j\le i$ 的位置。

可以直接在 attention score 上加入 mask：

$$
s_{ij}^{\mathrm{masked}}=
\begin{cases}
s_{ij}, & j\le i\\
-\infty, & j>i
\end{cases}
$$

再经过 softmax：

$$
\alpha_{ij}=\operatorname{softmax}(s_i^{\mathrm{masked}})_j
$$

由于 $\exp(-\infty)=0$，因此当 $j>i$ 时有 $\alpha_{ij}=0$，未来词元不会参与当前位置的计算。

因此，未来掩码使 Transformer 的 Self-Attention 满足自回归约束：每个位置只能使用自己以及之前的信息，不能查看未来。

![因果注意力掩码](/assets/img/posts/neural-network-architectures/causal-mask.png)

### 最简自注意力架构小结

一个最基本的 Self-Attention 模型可以概括为以下几个部分：

1. **输入表示**：将词元映射为词向量，并加入位置表示，使模型能够感知词元顺序。
2. **Self-Attention**：由输入计算 Query、Key 和 Value，通过注意力权重聚合整个序列的信息，得到上下文化表示。
3. **逐元素非线性**：在 Self-Attention 后加入 FFN，对每个词元独立进行非线性变换，提高模型的表示能力。
4. **未来掩码**：在自回归语言模型中加入 causal mask，使位置 $i$ 只能关注位置 $j\le i$ 的词元，避免使用未来信息。

因此，一个最简的自注意力架构可以表示为：

**词嵌入 + 位置表示 → Self-Attention → FFN → 输出**

若用于自回归语言模型，则在 Self-Attention 中额外加入**未来掩码**。

## Transformer
Transformer是一种基于自注意力的架构，由多个堆叠的块组成，每个块都包含自注意力层和前馈层，以及我们还将讨论的其他几个组件。下图给出了Transformer架构示意图。

![仅含解码器的 Transformer 架构](/assets/img/posts/neural-network-architectures/transformer-decoder.png)

### 多头自注意力

单头 Self-Attention 本质上是对所有 Value 进行一次加权平均。为了让模型能够同时从不同角度关注序列中的信息，可以并行执行多次 Self-Attention，这就是多头自注意力（Multi-Head Self-Attention）。

设共有 $k$ 个注意力头，每个头都有独立的参数矩阵 $Q^{(l)},K^{(l)},V^{(l)}\in\mathbb R^{d\times d/k}$。第 $l$ 个头首先计算自己的 Query、Key 和 Value，然后独立执行注意力：

$$
\alpha_{ij}^{(l)}=\frac{\exp\left((q_i^{(l)})^Tk_j^{(l)}\right)}{\sum_{j'=1}^{n}\exp\left((q_i^{(l)})^Tk_{j'}^{(l)}\right)}
$$

$$
h_i^{(l)}=\sum_{j=1}^{n}\alpha_{ij}^{(l)}v_j^{(l)}
$$

每个头的输出维度为 $d/k$。最后将所有头的结果拼接，再通过线性变换 $O\in\mathbb R^{d\times d}$ 得到最终表示：

$$
h_i=O[h_i^{(1)};\ldots;h_i^{(k)}]
$$

**Self-Attention 的矩阵形式**

设整个输入序列表示为 $X\in\mathbb R^{n\times d}$，则可以一次性计算所有词元的 Query、Key 和 Value：

$$
Q_X=XQ,\ K_X=XK,\ V_X=XV
$$

所有词元之间的注意力分数可以通过矩阵乘法同时得到：

$$
A=\operatorname{softmax}(Q_XK_X^T)\in\mathbb R^{n\times n}
$$

其中 $A_{ij}$ 表示位置 $i$ 对位置 $j$ 的注意力权重。最终对 Value 进行加权求和：

$$
H=AV_X=\operatorname{softmax}(Q_XK_X^T)V_X\in\mathbb R^{n\times d}
$$

下图展示了这些矩阵操作

![自注意力的矩阵形式](/assets/img/posts/neural-network-architectures/attention-matrix.png)

多头 Self-Attention 可以通过矩阵变形并行计算。先得到$XQ,XK,XV\in\mathbb R^{n\times d}$，设共有 $k$ 个头，将维度 $d$ 拆成 $k\times d/k$，并调整维度顺序：$Q,K,V\in\mathbb R^{k\times n\times d/k}$，于是 $k$ 个注意力头可以同时计算：$A=\operatorname{softmax}(QK^T),\ H=AV$，最后将各个头的输出重新拼接成 $d$ 维表示，再进行一次线性变换。
因此，多头注意力本质上是把一个 $d$ 维注意力拆成 $k$ 个 $d/k$ 维注意力并行计算，主要计算量与单头注意力相近

![多头注意力的并行矩阵计算](/assets/img/posts/neural-network-architectures/multi-head-attention-matrix.png)

### 层归一化

Transformer 中常使用层归一化来稳定每一层的激活，并改善训练时的梯度传播。

对于序列中位置 $i$ 的隐藏状态 $h_i\in\mathbb R^d$，LayerNorm 只在该词元自己的 $d$ 个隐藏维度上计算统计量，不会与其他位置混合。

首先计算均值和标准差：

$$
\hat{\mu}_i=\frac{1}{d}\sum_{j=1}^{d}h_{i,j},\ \hat{\sigma}_i=\sqrt{\frac{1}{d}\sum_{j=1}^{d}(h_{i,j}-\hat{\mu}_i)^2}
$$

然后进行归一化：

$$
\operatorname{LN}(h_i)=\frac{h_i-\hat{\mu}_i}{\hat{\sigma}_i}
$$

实际使用时通常还会加入可学习的缩放和偏置参数，使模型能够重新调整归一化后的表示。

它与 BatchNorm 的主要区别是：LayerNorm 不依赖 batch 中其他样本，因此很适合 Transformer 和序列模型。

### 残差连接

残差连接（Residual Connection）把某一层的输入直接加到该层输出上：

$$
h'=f(h)+h
$$

这样可以为梯度提供更直接的传播路径，使深层网络更容易训练。

在 Transformer 中，残差连接通常与 LayerNorm 一起使用，即 Add & Norm。常见有两种形式：

- **Pre-Norm**：先归一化，再经过子层并加入残差；

$$
h_{\text{pre-norm}}=f(\operatorname{LN}(h))+h
$$

- **Post-Norm**：先经过子层并加入残差，再归一化；

$$
h_{\text{post-norm}}=\operatorname{LN}(f(h)+h)
$$

其中 $f$ 可以是 Self-Attention，也可以是 FFN。实际中 Pre-Norm 通常具有更稳定的梯度传播。

### 注意力 logit 缩放

注意力使用 Query 和 Key 的点积衡量相关性。当维度 $d$ 增大时，点积的大小通常也会随之增大，使 softmax 的分布过于尖锐，影响梯度传播。

因此使用缩放点积注意力（Scaled Dot-Product Attention），将注意力分数除以 $\sqrt d$：

$$
A=\operatorname{softmax}\left(\frac{QK^T}{\sqrt d}\right)
$$

再对 Value 加权：

$$
H=AV=\operatorname{softmax}\left(\frac{QK^T}{\sqrt d}\right)V
$$

对于多头注意力，每个头的维度通常为 $d_k=d/k$，因此实际缩放因子为 $\sqrt{d_k}$。

### Transformer 编码器

Transformer Encoder 接收完整输入序列，因此不需要未来掩码。

首先将词元转换为词嵌入，并加入位置表示：

$$
H^{(0)}=X+P
$$

随后堆叠多个 Encoder Block。每个 Encoder Block 主要包含两个子层：

1. **Multi-Head Self-Attention**：让不同位置之间交换信息；
2. **FFN**：对每个位置独立进行非线性变换。

两个子层都配合残差连接和 LayerNorm。以 Pre-Norm 为例：

$$
H'=H+\operatorname{MHA}(\operatorname{LN}(H))
$$

$$
H_{\text{out}}=H'+\operatorname{FFN}(\operatorname{LN}(H'))
$$

多个 Encoder Block 依次堆叠后，最终得到每个词元的上下文化表示。

![Transformer 编码器架构](/assets/img/posts/neural-network-architectures/transformer-encoder.png)

Transformer Encoder 中没有未来掩码，因此每个位置都可以看到整个输入序列，适合需要获得完整序列上下文化表示的任务。

它的特点是：

- 每个词元都可以关注序列中的所有位置；
- 输出表示同时包含当前词元及其前后文信息；
- 更适合序列理解，而不是自回归文本生成。

### Transformer 解码器

Transformer Decoder 主要用于自回归语言建模。与 Encoder 的主要区别是：Decoder 中的 Self-Attention 使用未来掩码，使位置 $i$ 只能看到当前位置及之前的信息。

因此在生成第 $t$ 个词时，只能根据 $w_{1:t-1}$ 进行预测：

$$
w_t\sim\operatorname{softmax}(f(w_{1:t-1}))
$$

通过堆叠多个 Decoder Block，就可以构成自回归 Transformer。

### Transformer 编码器—解码器

Encoder-Decoder Transformer 同时接收两个序列。输入序列首先经过 Encoder，得到上下文化表示：

$$
H^{(x)}=\operatorname{TransformerEncoder}(x_{1:n})
$$

Decoder 一方面对目标序列进行带未来掩码的 Self-Attention，另一方面通过交叉注意力（Cross-Attention）读取 Encoder 的输出。

Cross-Attention 与 Self-Attention 的计算形式相同，区别在于 Query、Key 和 Value 来自不同序列：

$$
q_i=Qh_i^{(y)},\ k_j=Kh_j^{(x)},\ v_j=Vh_j^{(x)}
$$

其中 Query 来自 Decoder，Key 和 Value 来自 Encoder。注意力权重为

$$
\alpha_{ij}=\frac{\exp(q_i^Tk_j)}{\sum_{j'=1}^{n}\exp(q_i^Tk_{j'})}
$$

再对 Encoder 的 Value 加权求和：

$$
h_i=\sum_{j=1}^{n}\alpha_{ij}v_j
$$

![Transformer 编码器—解码器架构](/assets/img/posts/neural-network-architectures/transformer-encoder-decoder.png)

当任务既需要双向上下文理解，又需要自回归生成输出时，可以使用 Encoder-Decoder 架构。

其中：

- Encoder 可以让每个输入词元关注整个输入序列，从而获得完整的上下文化表示；
- Decoder 根据已经生成的内容逐步预测后续输出；
- Cross-Attention 使 Decoder 在生成过程中能够读取 Encoder 的表示。

Encoder-Decoder 很适合输入序列到输出序列的任务，例如摘要、翻译等。

不过这种架构需要在 Encoder 和 Decoder 之间分配模型参数。随着模型规模增大，仅使用 Decoder 的架构通常更加简单，因此许多大规模语言模型采用 Decoder-only Transformer。
