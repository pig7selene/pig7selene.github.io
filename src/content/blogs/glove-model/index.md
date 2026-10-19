---
title: "CS224N | GloVe 模型"
publishDate: 2026-08-06
category: learning
tags: [NLP, CS224N]
language: zh
description: "CS224n lec2,the GloVe model, co-occurrence matrices, least-squares objectives, and the evaluation of word vectors."
heroImage:
  src: /assets/img/posts/neural-network-architectures/transformer-card-cover.png
  alt: "文章封面"
  color: "#9a7093"
---

<aside class="download-callout" role="note">
  <p>You can download the <a href="/output/pdf/glove-model.tex">LaTeX</a> and <a href="/output/pdf/glove-model.pdf">PDF</a>.</p>
</aside>

# GloVe 模型

## 1.1 与之前模型的比较

之前我们学习了 skip-gram 模型，它主要通过在局部上下文窗口中进行预测来学习词嵌入。它能捕获超越简单词语相似度的复杂语言规律，但没有利用全局共现统计信息。

GloVe 则利用全局统计信息，通过最小二乘目标预测词 $j$ 出现在词 $i$ 上下文中的概率。

## 1.2 共现矩阵

设 $X$ 为词—词共现矩阵，其中 $X_{ij}$ 表示词 $j$ 出现在词 $i$ 的上下文中的次数。

令

$$
X_i = \sum_k X_{ik}
$$

表示任意词 $k$ 出现在词 $i$ 的上下文中的总次数。

最后，令

$$
P_{ij}=P(w_j \mid w_i)=\frac{X_{ij}}{X_i}
$$

表示词 $j$ 出现在词 $i$ 的上下文中的概率。

## 1.3 最小二乘目标函数

在 skip-gram 模型中，我们通过 softmax 计算词 $j$ 出现在词 $i$ 的上下文中的概率：

$$
Q_{ij} = \frac{exp(u_{j}^{T}x_i)}{\sum_{w=1}^{W}exp(u_w^T v_i)}
$$

训练以在线、随机的方式进行，对应的全局交叉熵损失为：

$$
J = -\sum_{j \in corpus}\sum_{j \in context(i)} logQ_{ij}
$$

交叉熵的一个显著缺点是要求分布 $Q$ 正确归一化，而归一化需要对整个词表进行代价较高的求和。因此，我们改用最小二乘目标，并去掉 $P$ 和 $Q$ 中的归一化因子：

$$
\hat{J}=\sum_{i=1}^{W}\sum_{j=1}^{W}X_i\left(\hat{p}_{ij}-\hat{Q}_{ij}\right)^2
$$

其中

$$
\hat{P}_{ij}=X_{ij},\qquad
\hat{Q}_{ij}=\exp\left(u_j^{T}v_i\right)
$$

二者都是未归一化分布。由于 $X_{ij}$ 往往很大，优化比较困难。一种有效改进是最小化 $\hat{P}$ 与 $\hat{Q}$ 的对数之间的平方误差：

$$
\hat{J}=\sum_{i=1}^{W}\sum_{j=1}^{W}X_i\left(\log\hat{P}_{ij}-\log\hat{Q}_{ij}\right)^2=\sum_{i=1}^{W}\sum_{j=1}^{W}X_i\left(u_j^{T}v_i-\log X_{ij}\right)^2
$$

# 词向量的评估

## 2.1 内在评估

词向量的内在评估，是在一个具体的中间子任务上评估某种嵌入方法生成的词向量。这类子任务通常比较简单、计算较快，可以帮助我们理解生成词向量的系统。内在评估一般返回一个数值，用于衡量词向量在该子任务上的表现。

## 2.2 外在评估

词向量的外在评估，是在实际需要解决的任务上评估词向量。这类任务通常更复杂、计算更慢。一般来说，仅凭一个表现不佳的外在评估系统，很难判断究竟是哪个具体子系统出了问题，这也说明了内在评估的必要性。
