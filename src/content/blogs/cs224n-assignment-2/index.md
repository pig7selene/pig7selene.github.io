---
title: "CS224N | 作业 2"
publishDate: 2026-08-25
category: learning
tags: [NLP, CS224N]
language: zh
description: "personal solution to CS224N assignment 2"
heroImage:
  src: /assets/img/posts/neural-network-architectures/transformer-card-cover.png
  alt: "文章封面"
  color: "#9a7093"
---

<aside class="download-callout" role="note">
  <p>You can download the <a href="/output/pdf/cs224n-assignment-2.tex">LaTeX</a> and <a href="/output/pdf/cs224n-assignment-2.pdf">PDF</a>.</p>
</aside>

## Problem 1

### (a)

设 $y \in \mathbb{R}^{\lvert V\rvert}$ 为真实分布对应的 one-hot 向量，$\hat{y} \in \mathbb{R}^{\lvert V\rvert}$ 为预测分布，并令

$$
\hat{y}_w = P(O=w\mid C=c) = \frac{\exp(u_w^\top v_c)} {\sum_{j\in V}\exp(u_j^\top v_c)}.
$$

矩阵 $U$ 的每一列是一个上下文词向量：

$$
U = \begin{bmatrix}u_1 & u_2 & \cdots & u_{\lvert V\rvert}\end{bmatrix}.
$$

由于 $y$ 是 one-hot 向量，$y_o=1$，而所有 $w\neq o$ 都满足 $y_w=0$。因此，

$$
-\sum_{w\in V}y_w\log \hat{y}_w = -\log \hat{y}_o.
$$

### (b)

**(i)**

先将朴素 softmax 的损失改写为

$$
J_{\text{naive-softmax}}(v_c,o,U) = -\log \hat{y}_o = -u_o^\top v_c+ \log\left(\sum_{w\in V}\exp(u_w^\top v_c)\right).
$$

对 $v_c$ 求导，得到

$$
\frac{\partial J}{\partial v_c} = -u_o+ \frac{\sum_{w\in V}\exp(u_w^\top v_c)u_w} {\sum_{j\in V}\exp(u_j^\top v_c)} = -u_o + \sum_{w\in V}\hat{y}_w u_w.
$$

由于 $y$ 是 one-hot 向量，

$$
u_o = Uy,\quad \sum_{w\in V}\hat{y}_w u_w = U\hat{y}.
$$

因此，向量化形式为

$$
\frac{\partial J}{\partial v_c}=U(\hat{y}-y)
$$

**(ii)**

梯度为零，当且仅当

$$
U(\hat{y}-y)=0
$$

等价地，$\hat{y}-y\in\operatorname{Null}(U)$。特别地，$\hat{y}=y$ 是梯度为零的充分条件。

**(iii)**

取学习率 $\eta>0$，梯度下降更新为

$$
v_c \leftarrow v_c-\eta\left(U\hat{y}-Uy\right) =v_c+\eta u_o-\eta\sum_{w\in V}\hat{y}_w u_w.
$$

其中，$+\eta u_o$ 项把中心词向量 $v_c$ 拉向真实上下文词向量 $u_o$，提高两者的相似度。$-\eta\sum_w\hat{y}_w u_w$ 项则让 $v_c$ 远离按预测概率加权的上下文词向量均值；模型当前赋予概率越高的词，受到的修正越大。两项共同提高真实上下文词相对于其他词的得分。

### (c)

设 $x\neq y$ 且 $u_x=\alpha u_y$。当 $\alpha>0$ 时，

$$
\frac{u_x}{\lVert u_x\rVert_2} =\frac{\alpha u_y}{\lVert \alpha u_y\rVert_2} =\frac{u_y}{\lVert u_y\rVert_2}.
$$

因此，L2 归一化会让方向相同、模长不同的两个向量变得完全相同。如果模长包含与短语分类有关的信息，例如 $x$ 和 $y$ 的语义方向相近，但在分类器中的强度或倾向不同，就会丢失有用信息。如果任务只关心方向，或者同方向、不同模长的向量本来就应产生相同分类效果，则不会丢失有用信息。

### (d)

由于 $U$ 由上下文词向量按列组成，
它的梯度也具有相同的列结构：

$$
\frac{\partial J(v_c,o,U)}{\partial U}= \begin{bmatrix} \dfrac{\partial J(v_c,o,U)}{\partial u_1} & \dfrac{\partial J(v_c,o,U)}{\partial u_2} & \cdots & \dfrac{\partial J(v_c,o,U)}{\partial u_{\lvert V\rvert}} \end{bmatrix}
$$

### (e)

损失为

$$
J=-u_o^\top v_c+\log\left(\sum_{j\in V}\exp(u_j^\top v_c)\right).
$$

对于真实上下文词 $w=o$，

$$
\frac{\partial J}{\partial u_o} =-v_c+\frac{\exp(u_o^\top v_c)}{\sum_{j\in V}\exp(u_j^\top v_c)}v_c =(\hat{y}_o-1)v_c.
$$

对于所有 $w\neq o$，

$$
\frac{\partial J}{\partial u_w} =\frac{\exp(u_w^\top v_c)}{\sum_{j\in V}\exp(u_j^\top v_c)}v_c =\hat{y}_w v_c.
$$

因此，

$$
\frac{\partial J}{\partial u_w} =(\hat{y}_w-y_w)v_c= \begin{cases} (\hat{y}_o-1)v_c, & w=o,\\\ \hat{y}_w v_c, & w\neq o. \end{cases}
$$

### (f)

对于 $f(x)=\max(\alpha x,x)$，其中 $0<\alpha<1$，

$$
f'(x)= \begin{cases} 1, & x>0,\\\ \alpha, & x<0. \end{cases}
$$

本题不要求讨论 $x=0$ 处的导数。

### (g)

从

$$
\sigma(x)=\frac{1}{1+e^{-x}},
$$

可得

$$
\sigma'(x) =\frac{e^{-x}}{(1+e^{-x})^2} =\frac{1}{1+e^{-x}}\left(1-\frac{1}{1+e^{-x}}\right).
$$

因此，

$$
\sigma'(x)=\sigma(x)\bigl(1-\sigma(x)\bigr)
$$

## Problem 2

### (a)

**(i)**

Momentum 对多个小批次的梯度进行平均，单个小批次梯度中的随机波动往往会相互抵消。持续有用的方向会在 $m$ 中累积，使参数更新更平滑、噪声更小，从而减少振荡，让模型更稳定、更快地收敛。

**(ii)**

历史梯度较小的参数对应的 $v$ 较小，因此会获得相对更大的更新；历史梯度持续较大的参数则获得较小的更新。这种自适应缩放既避免大梯度参数变化过猛，也让小梯度参数能够取得有效进展。

### (b)

**(i)**

对于每个隐藏单元 $i$，

$$
(h_{\text{drop}})_i=\gamma d_i h_i.
$$

由于 $\mathbb{E}[d_i]=1-p_{\text{drop}}$，

$$
\mathbb{E}[(h_{\text{drop}})_i]=
\gamma h_i(1-p_{\text{drop}}).
$$

要求 $\mathbb{E}[(h_{\text{drop}})_i]=h_i$，可得

$$
\gamma=\frac{1}{1-p_{\text{drop}}}.
$$

**(ii)**

训练时使用 Dropout 进行正则化，随机丢弃隐藏单元能够防止模型过度依赖特定神经元，从而减少过拟合。评估时关闭 Dropout，使用完整网络，使预测保持稳定和确定。

## Problem 3

### (a)

所需的转移序列如下：

| 栈 | 缓冲区 | 新增依存关系 | 转移动作 |
| --------------------------------------------- | -------------------------------------------------------- | ------------------------ | --------------------- |
| `[ROOT]`                                      | `[I, presented, my, findings, at, the, NLP, conference]` |                          | 初始配置 |
| `[ROOT, I]`                                   | `[presented, my, findings, at, the, NLP, conference]`    |                          | SHIFT                 |
| `[ROOT, I, presented]`                        | `[my, findings, at, the, NLP, conference]`               |                          | SHIFT                 |
| `[ROOT, presented]`                           | `[my, findings, at, the, NLP, conference]`               | `presented → I`          | LEFT-ARC              |
| `[ROOT, presented, my]`                       | `[findings, at, the, NLP, conference]`                   |                          | SHIFT                 |
| `[ROOT, presented, my, findings]`             | `[at, the, NLP, conference]`                             |                          | SHIFT                 |
| `[ROOT, presented, findings]`                 | `[at, the, NLP, conference]`                             | `findings → my`          | LEFT-ARC              |
| `[ROOT, presented]`                           | `[at, the, NLP, conference]`                             | `presented → findings`   | RIGHT-ARC             |
| `[ROOT, presented, at]`                       | `[the, NLP, conference]`                                 |                          | SHIFT                 |
| `[ROOT, presented, at, the]`                  | `[NLP, conference]`                                      |                          | SHIFT                 |
| `[ROOT, presented, at, the, NLP]`             | `[conference]`                                           |                          | SHIFT                 |
| `[ROOT, presented, at, the, NLP, conference]` | `[]`                                                     |                          | SHIFT                 |
| `[ROOT, presented, at, the, conference]`      | `[]`                                                     | `conference → NLP`       | LEFT-ARC              |
| `[ROOT, presented, at, conference]`           | `[]`                                                     | `conference → the`       | LEFT-ARC              |
| `[ROOT, presented, conference]`               | `[]`                                                     | `conference → at`        | LEFT-ARC              |
| `[ROOT, presented]`                           | `[]`                                                     | `presented → conference` | RIGHT-ARC             |
| `[ROOT]`                                      | `[]`                                                     | `ROOT → presented`       | RIGHT-ARC             |

### (b)

包含 $n$ 个词的句子需要 $2n$ 次转移。每个词恰好经过一次 SHIFT 入栈，并通过一次 LEFT-ARC 或 RIGHT-ARC 从栈中移除。

### (c)

<img src="/assets/img/posts/cs224n-assignment-2/code2.png" alt="code2" style="zoom:50%;" />

![code](/assets/img/posts/cs224n-assignment-2/code.png)

### (d)

`minibatch_parse` 为每个输入句子创建一个 `PartialParse` 对象，并保留原始顺序。每轮选取至多 `batch_size` 个未完成的分析对象，让模型分别预测一个转移动作，再执行这些动作。只有当缓冲区为空且栈中只剩 `ROOT` 时，才将该对象从未完成列表中移除。最后从原始对象列表中收集依存关系，因而保留输入句子的顺序。

![](/assets/img/posts/cs224n-assignment-2/code3.png)

### (e)

**(i)**

令

$$
\mathbf{z} = \mathbf{xW} + \mathbf{b}_1,
\qquad
\mathbf{h} = \operatorname{ReLU}(\mathbf{z}).
$$

对于单个隐藏单元 $h_i$，

$$
h_i = \max(z_i, 0)
= \max\left(\sum_k x_k W_{ki} + b_{1,i}, 0\right).
$$

因此，对于输入特征 $x_j$，

$$
\begin{cases}
W_{ji}, & z_i > 0,\\
0, & z_i < 0.
\end{cases}
$$

在 $z_i=0$ 处导数不存在，题目允许忽略这一点。

**(ii)**

设 logits 为 $\mathbf{l}$，并令

$$
\hat{y}_j = \frac{e^{l_j}}{\sum_k e^{l_k}}
$$

为类别 $j$ 的 softmax 概率。交叉熵损失为

$$
-\sum_j y_j \log \hat{y}_j.
$$

对于正确类别为 $c$ 的 one-hot 目标向量，损失关于 logit $l_i$ 的导数为

$$
\hat{y}_i - y_i
$$

等价地，

$$
\begin{cases}
\hat{y}_i - 1, & i=c,\\
\hat{y}_i, & i\neq c.
\end{cases}
$$

训练时，这个梯度会提高正确转移动作的 logit，并降低模型对错误转移动作的相对偏好。

**(iii) UAS**

![](/assets/img/posts/cs224n-assignment-2/screenshot-2026-09-14-19-40-52.png)

![](/assets/img/posts/cs224n-assignment-2/screenshot-2026-09-14-19-39-50.png)

模型在开发集上取得的最佳 UAS 为 **88.85**。使用开发集上的最佳检查点恢复的最终模型，在测试集上取得了 **89.23** 的 UAS。

UAS 衡量预测句法中心词正确的词所占比例，不考虑依存关系标签。

**我的代码**

![__init__](/assets/img/posts/cs224n-assignment-2/code4.png)

![](/assets/img/posts/cs224n-assignment-2/code5.png)

![](/assets/img/posts/cs224n-assignment-2/code6.png)

**测试**

![](/assets/img/posts/cs224n-assignment-2/screenshot-2026-09-14-19-21-52.png)
