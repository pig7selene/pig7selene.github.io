---
title: "CS224N | Word2Vec 模型"
publishDate: 2026-08-03
category: learning
tags: [NLP, CS224N]
language: zh
description: "CS224n lec1,word representation, distributional semantics, and the Skip-gram Word2Vec model"
heroImage:
  src: /assets/img/posts/neural-network-architectures/transformer-card-cover.png
  alt: "文章封面"
  color: "#9a7093"
---

<aside class="download-callout" role="note">
  <p>You can download the <a href="/output/pdf/word2vec-model.tex">LaTeX</a> and <a href="/output/pdf/word2vec-model.pdf">PDF</a>.</p>
</aside>

## 词表示

### 1.1 能指与所指

词语是能指，代表现实或想象世界中的某个所指对象。能指及其所指可以非常宽泛，例如“叔叔”和“饮料”，也可以更加具体，例如“祖寇叔叔”和“咖啡”。因此，词义十分重要，即使很小的变化也可能带来很大影响。

### 1.2 独立的词与向量

表示词语最简单的方法，是将它们视为相互独立、互不相关的实体。相互独立的分量通常用 one-hot 向量，也就是标准基向量表示。例如，

$$
v_{\mathrm{tea}}=[0,0,1,0,\ldots,0]^T,
\qquad
v_{\mathrm{coffee}}=[0,0,0,0,\ldots,1,\ldots,0]^T.
$$

但 one-hot 表示无法编码词语之间的相似关系：任意两个不同词向量的点积都为 $0$。而实际句子中的词语彼此有关，因此需要引入下面的方法。

### 1.3 用人工标注的离散属性构造向量

这种方法手工构造特征向量，用各个分量描述词语的属性。显然，这并不是一个理想的解决方案。

## 分布语义学与 Word2Vec

### 2.1 分布假设

分布假设认为，可以从一个词出现的上下文分布中推断其含义。这是 NLP 中最有影响力的思想之一。

### 2.2 共现矩阵与文档上下文

为实现分布假设的思想，可以构造一个共现矩阵，记为 $X$。矩阵大小为 $\lvert V\rvert\times\lvert V\rvert$，其中 $X_{ij}$ 表示词 $i$ 与词 $j$ 出现在相同上下文中的次数。

在一个句子中，可以划定不同大小的窗口。较短的窗口倾向于编码句法性质，较长的窗口则倾向于编码语义性质，甚至是主题层面的性质。

这种方法的问题在于，高维向量不便于现代神经网络系统处理，而且高频词会获得过大的权重。

### 2.3 Word2Vec 模型与目标函数

Word2Vec 将固定词表中的每个词表示为一个低维向量，维度远小于词表大小。它学习词向量的数值，使模型能够通过一个简单函数预测上下文词的分布。这里介绍的是 **Skip-gram Word2Vec**。

**Skip-gram Word2Vec**

设 $C,O$ 是一对随机变量，$C \in V$ 表示中心词，$O \in V$ 表示出现在中心词上下文中的外部词，用 $c,o$ 表示它们的具体取值。

设词表为 $V$。每个词有两种 $d$ 维向量表示：

- $u_w$：词 $w$ 作为上下文词时的向量；
- $v_w$：词 $w$ 作为中心词时的向量。

将所有词向量排列成矩阵

$$
U,V\in\mathbb{R}^{\lvert V\rvert\times d}.
$$

给定中心词 $c$，上下文词为 $o$ 的概率定义为

$$
p_{U,V}(o\mid c)=
\frac{\exp(u_o^\top v_c)}
{\sum_{w\in V}\exp(u_w^\top v_c)}.
$$

这里，$u_o^\top v_c$ 表示两个词向量的相似度。点积越大，模型就越认为词 $o$ 容易出现在词 $c$ 的上下文中。

Softmax 将所有词的点积得分转换为概率分布：

$$
p_{U,V}(\cdot\mid c)\in\mathbb R^{\lvert V\rvert}.
$$

训练 Word2Vec 的目标，是让预测概率分布接近语料中词 $c$ 的真实上下文词分布，也就是共现矩阵中 $c$ 对应的那一行。

到目前为止，我们只定义了模型，还需要通过训练确定参数 $U,V$。Word2Vec 使用交叉熵损失，让预测的条件分布接近真实上下文分布：

$$
\min_{U,V}
\mathbb E_{o,c}
\left[
-\log p_{U,V}(o\mid c)
\right].
$$

这里，$(o,c)$ 表示从语料中采样的一对上下文词和中心词。对每个样本，模型计算给定中心词 $c$ 时，真实上下文词 $o$ 的负对数概率：

$$
-\log p_{U,V}(o\mid c).
$$

模型赋予真实上下文词的概率越大，损失越小。因此训练会持续调整 $U,V$，提高真实词对的预测概率。

### 2.4 从语料中估计 Word2Vec

**Word2Vec 的经验损失**

设 $D$ 是由文档 $\{d\}$ 组成的集合。每篇文档是词序列 $w_1^{(d)},\dots,w_m^{(d)}$，其中 $w^{(d)} \in V$。设 $k \in \mathbb N_{++}$ 为正整数窗口大小，现在将随机变量 $C,O$ 与这个具体数据集联系起来。

中心词 $C$ 依次取文档中的每个词 $w_i$。对每个 $w_i$，其上下文词为

$$
\{w_{i-k},\dots,w_{i-1},w_{i+1},\dots,w_{i+k}\}.
$$

因此，目标函数变为

$$
L(U,V)=
\sum_{d\in D}
\sum_{i=1}^{m}
\sum_{\substack{j=-k\\j\ne0}}^{k}
-\log p_{U,V}
\left(
w_{i+j}^{(d)}
\mid
w_i^{(d)}
\right).
$$

这里对所有文档、各文档中的所有词，以及窗口内的所有词求和，累积给定中心词时上下文词的负对数似然。

**基于梯度的估计**

先在信息很少的情况下为 $U,V$ 猜测一个初值，再不断沿着局部最有利于改进这个猜测的方向移动，由此得到基于梯度的实现。

对于标量函数 $f$，关于参数矩阵 $U$ 的梯度记为 $\nabla_U f$。梯度给出函数上升最快的方向，因此要最小化损失，就应沿梯度的反方向更新参数。

训练开始前，通常将词向量矩阵 $U,V$ 初始化为很小的随机数：

$$
U^{(0)},V^{(0)}\sim N(0,0.001)^{\lvert V\rvert\times d}.
$$

也就是说，矩阵中的每个元素都独立采样自均值为 $0$、方差较小的正态分布。之后使用梯度下降反复更新参数，例如矩阵 $U$ 的更新公式为

$$
U^{(i+1)}=
U^{(i)}-
\alpha\nabla_U L\left(U^{(i)},V^{(i)}\right).
$$

其中，$\alpha$ 为学习率，控制每次参数更新的步长。矩阵 $V$ 也以相同方式更新：

$$
V^{(i+1)}=
V^{(i)}-
\alpha\nabla_V L\left(U^{(i)},V^{(i)}\right).
$$

不断沿降低损失的方向更新 $U,V$ 后，模型的预测分布就会逐渐接近真实上下文词分布。

**随机梯度**

这里暂不讨论如何显式计算梯度。计算 $L(U,V)$ 需要遍历整个训练集，代价很高，因此使用随机梯度优化。

### 2.5 完整的梯度推导

先写出梯度，再利用微分的线性性质将梯度算子移入求和：

$$
\nabla_{v_c}\widehat L(U,V)=
\sum_{d\in D}
\sum_{i=1}^{m}
\sum_{\substack{j=-k\\j\ne 0}}^{k}
-\nabla_{v_c}
\log p_{U,V}
\left(
w_{i+j}^{(d)}
\mid
w_i^{(d)}
\right).
$$

将 $w_{i+j}^{(d)}$ 简记为 $o$，将 $w_i^{(d)}$ 简记为 $c$：

$$
\begin{aligned}
\nabla_{v_c}\log p_{U,V}(o\mid c)
&=
\nabla_{v_c}\log
\frac{\exp(u_o^\top v_c)}
{\sum_{w\in V}\exp(u_w^\top v_c)} \\
&=
\nabla_{v_c}\log\exp(u_o^\top v_c)
-
\nabla_{v_c}\log
\sum_{w\in V}\exp(u_w^\top v_c).
\end{aligned}
$$

分别对这两部分关于 $v_c$ 求梯度。

第一部分为

$$
\nabla_{v_c}\log\exp(u_o^\top v_c)=
\nabla_{v_c}(u_o^\top v_c)=
u_o.
$$

第二部分为

$$
\begin{aligned}
\nabla_{v_c}
\log\sum_{w\in V}\exp(u_w^\top v_c)
&=
\frac{
\sum_{x\in V}
\exp(u_x^\top v_c)u_x
}{
\sum_{w\in V}
\exp(u_w^\top v_c)
} \\
&=
\sum_{x\in V}
p_{U,V}(x\mid c)u_x.
\end{aligned}
$$

因此，

$$
\nabla_{v_c}\log p_{U,V}(o\mid c)=
u_o-
\sum_{x\in V}
p_{U,V}(x\mid c)u_x.
$$

其中，$u_o$ 是实际观测到的上下文词向量，而

$$
\sum_{x\in V}p_{U,V}(x\mid c)u_x
$$

是模型预测分布下上下文词向量的期望。因此，这个梯度可以理解为“观测值减去期望值”。

对于负对数似然损失，梯度方向相反：

$$
\nabla_{v_c}\left[-\log p_{U,V}(o\mid c)\right]=
\sum_{x\in V}
p_{U,V}(x\mid c)u_x-u_o.
$$

梯度下降使中心词向量 $v_c$ 更接近真实上下文词向量 $u_o$，同时远离那些被模型错误赋予高概率的词向量。
