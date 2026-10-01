---
title: "CS224N | Assignment 3"
publishDate: 2026-10-01
category: learning
tags: [NLP, CS224N, Transformer]
language: zh
description: "CS224N Assignment 3 notes on self-attention, multi-head attention, permutation equivariance, and positional embeddings."
heroImage:
  src: /assets/img/posts/neural-network-architectures/transformer-card-cover.png
  alt: "Sunset cityscape with a figure looking over the skyline"
  color: "#9a7093"
---

## Problem 1

### (a)

**1.** 让 $k_j^\top q$ 比所有其他 $k_i^\top q$ 大得多即可。

**2.** 约等于 $v_j$。

### (b)

设 $q=\lambda(k_a+k_b)$，$\lambda$ 是一个足够大的正数。因为 key 两两正交且都是单位向量，易得

$$
k_a^\top q = k_b^\top q = \lambda
$$

以及对于 $i\ne a,b$，有 $k_i^\top q=0$。因此 $k_a,k_b$ 的 softmax score 都为 $e^{\lambda}$，其余的都为 $e^0$。当 $\lambda$ 足够大的时候，可以忽略其余 key 的贡献，因此

$$
c=\sum_i\alpha_i v_i \approx\frac12(v_a+v_b)
$$

### (c)

**1.** 和 (b) 类似，只是把确定的实际采样出来的 $k$ 变成了均值，剩下的思路和 (b) 一样。因为扰动项很少可以忽略，所以均值仍然可以看作 $k_a$ 和 $k_b$。

**2.** 这个题专门让 $k_a$ 沿着 $\mu_a$ 方向大幅伸缩，可以看成 $k_a \approx s\mu_a$，其中 $s$ 有很大波动。因此我们可以得到 $k_a^\top q\approx\lambda s$，而 $k_b$ 模长比较稳定，因此 $k_b^\top q\approx\lambda$。这导致输出 $c$ 会在 $v_a$ 和 $v_b$ 之间摇摆；第一题中 $c$ 的方差很小，比较稳定，而第二题中 $c$ 的方差较大。

### (d)

**1.** 这题区别于 (c)：上题是让单头用一个 query 同时关注 $a$ 和 $b$，这题是让两个 head 分工。因为各 $\mu_i$ 两两正交，且 key 只有极小扰动，$q_1$ 只会与 $k_a$ 有较大点积，因此 $c_1\approx v_a$；同理 $c_2\approx v_b$，故

$$
c=\frac12(c_1+c_2) \approx\frac12(v_a+v_b)
$$

**2.** 类似于 (c)(ii) 的多头版本。$k_a$ 的模长波动会使 $k_a^\top q_1$ 波动，因此 $c_1$ 的方差较大；$k_a$ 较长时 $c_1\approx v_a$，较短时对 $v_a$ 的关注减弱。但 $q_2\perp\mu_a$，故 $k_a$ 沿 $\mu_a$ 方向的模长波动几乎不影响第二个 head，仍有 $c_2\approx v_b$ 且 $c_2$ 方差很小。因此：

$$
c=\frac12(c_1+c_2)
$$

只有第一个 head 的输出波动，第二个 head 保持稳定。相比 (c)(ii) 的单头 Attention 中 $v_a,v_b$ 的权重会同时波动，多头 Attention 的最终输出 $c$ 方差更小。

### (e)

多头 Attention 将原本由单个 head 同时关注 $v_a,v_b$ 的任务拆分开：一个 head 关注 $v_a$，另一个 head 关注 $v_b$。因此某个 key 的模长波动只会影响对应 head 的输出，其他 head 仍保持稳定，最终再对各 head 输出求平均，能降低输出方差，使 Attention 更鲁棒。

## Problem 2

### (a)

**1.** 由 $X_{\mathrm{perm}}=PX$ 可得 $Q_{\mathrm{perm}}=PQ,K_{\mathrm{perm}}=PK,V_{\mathrm{perm}}=PV$，因此：

$$
\frac{Q_{\mathrm{perm}}K_{\mathrm{perm}}^\top}{\sqrt d}
=
P\frac{QK^\top}{\sqrt d}P^\top
$$

由题设 Softmax 性质及 $P^\top P=I$：

$$
\begin{aligned}
H_{\mathrm{perm}}
&=\operatorname{softmax}\left(
P\frac{QK^\top}{\sqrt d}P^\top
\right)PV \\
&=P\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt d}
\right)P^\top PV \\
&=PH
\end{aligned}
$$

又因为 $P\mathbf1=\mathbf1$，故：

$$
\begin{aligned}
Z_{\mathrm{perm}}
&=\operatorname{ReLU}(PHW_1+\mathbf1b_1)W_2+\mathbf1b_2 \\
&=\operatorname{ReLU}\left(P(HW_1+\mathbf1b_1)\right)W_2+P\mathbf1b_2 \\
&=P\operatorname{ReLU}(HW_1+\mathbf1b_1)W_2+P\mathbf1b_2 \\
&=PZ
\end{aligned}
$$

**2.** Transformer 不加入 Position Embedding 时满足置换等变性 $Z_{\mathrm{perm}}=PZ$，模型无法感知 token 原本所在的位置。但文本语义依赖词序，因此该性质会使 Transformer 无法区分不同词序的文本，需要加入 Position Embedding 来提供位置信息。

### (b)

**1.** 能解决。加入 Position Embedding 后 $X_{\mathrm{pos}}=X+\Phi$，若只置换 token 而位置不变，输入变为 $X_{\mathrm{perm,pos}}=PX+\Phi$，而原输入整体置换为 $P(X+\Phi)=PX+P\Phi$，因此不同词序会产生不同的输入表示。每个位置具有不同的 Position Embedding，Transformer 可以据此感知 token 的位置和顺序。

**2.** 不会。若两个不同位置 $t\ne s$ 的 Position Embedding 相同，则其前两维必须满足 $\sin t=\sin s,\cos t=\cos s$，因此 $t=s+2\pi k$，其中 $k$ 为整数。由于 $t,s$ 均为整数，而 $2\pi k$ 仅在 $k=0$ 时为整数，故只能有 $t=s$。
