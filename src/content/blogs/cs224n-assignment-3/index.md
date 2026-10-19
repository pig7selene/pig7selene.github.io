---
title: "CS224N | 作业 3"
publishDate: 2026-10-01
category: learning
tags: [NLP, CS224N, Transformer]
language: zh
description: "Personal solutions to CS224N Assignment 3: attention, permutation equivariance, positional embeddings, and a decoder-only Transformer."
heroImage:
  src: /assets/img/posts/neural-network-architectures/transformer-card-cover.png
  alt: "文章封面"
  color: "#9a7093"
---

<aside class="download-callout" role="note">
  <p>You can download the <a href="/output/pdf/cs224n-assignment-3.tex">LaTeX</a> and <a href="/output/pdf/cs224n-assignment-3.pdf">PDF</a>.</p>
</aside>

## Problem 1

### (a)

**1.** 让 $k_j^\top q$ 比所有其他 $k_i^\top q$ 大得多即可
**2.** 约等于 $v_j$

### (b)
设 $q=\lambda(k_a+k_b)$， $\lambda$ 是一个足够大的正数，因为key两两正交且都是单位向量，易得

$$
k_a^\top q = k_b^\top q =\lambda
$$

以及对于 $i\ne a,b$，有 $k_i^\top q  =0$，因此 $k_a,k_b$ 的softmax score都为 $e^{\lambda}$，其余的都为 $e^0$，当 $\lambda$ 足够大的时候可以忽略其余key的贡献，因此

$$
c=\sum_i\alpha_i v_i \approx\frac12(v_a+v_b)
$$

### (c)
**1.** 和 (b) 类似，只是把确定的实际采样出来的 $k$ 变成了均值，剩下的思路和 (b) 一样，因为扰动项很少可以忽略，所以均值仍然可以看作 $k_a$ 和 $k_b$
**2.** 这个题专门让 $k_a$ 沿着 $\mu_a$ 方向大幅伸缩，可以看成 $k_a \approx s \mu_a$，其中，$s$ 有很大波动，因此我们可以得到 $k_a^\top q\approx\lambda s$，而 $k_b$ 模长比较稳定，因此 $k_b^\top q\approx\lambda$，所以导致了输出 $c$ 会在 $v_a$ 和 $v_b$ 之间摇摆，而第一题中 $c$ 的方差很小，比较稳定；而第二题中 $c$ 的方差较大

### (d)
**1.** 这题区别于 (c) ，上题是让单头用一个query同时关注a和b，这题是让两个head分工，因为各 $\mu_i$ 两两正交，且 key 只有极小扰动，$q_1$ 只会与 $k_a$ 有较大点积，因此 $c_1\approx v_a$，同理 $c_2\approx v_b$，故

$$
c=\frac12(c_1+c_2) \approx\frac12(v_a+v_b)
$$

**2.** 类似于 (c)(ii) 的多头版本，$k_a$ 的模长波动会使 $k_a^\top q_1$ 波动，因此 $c_1$ 的方差较大，$k_a$ 较长时 $c_1\approx v_a$，较短时对 $v_a$ 的关注减弱。但 $q_2\perp\mu_a$，故 $k_a$ 沿 $\mu_a$ 方向的模长波动几乎不影响第二个 head，仍有 $c_2\approx v_b$ 且 $c_2$ 方差很小。因此：

$$
c=\frac12(c_1+c_2)
$$

只有第一个 head 的输出波动，第二个 head 保持稳定。相比 (c)(ii) 的单头 Attention 中 $v_a,v_b$ 的权重会同时波动，多头 Attention 的最终输出 $c$ 方差更小

### (e)
多头 Attention 将原本由单个 head 同时关注 $v_a,v_b$ 的任务拆分开：一个 head 关注 $v_a$，另一个 head 关注 $v_b$。因此某个 key 的模长波动只会影响对应 head 的输出，其他 head 仍保持稳定，最终再对各 head 输出求平均，能降低输出方差，使 Attention 更鲁棒

## Problem 2

### (a)
**1.** 由 $X_{\mathrm{perm}}=PX$ 可得 $Q_{\mathrm{perm}}=PQ,K_{\mathrm{perm}}=PK,V_{\mathrm{perm}}=PV$ ，因此：

$$
\frac{Q_{\mathrm{perm}}K_{\mathrm{perm}}^\top}{\sqrt d}
=
P\frac{QK^\top}{\sqrt d}P^\top
$$

由题设 Softmax 性质及 $P^\top P=I$

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

**2.** Transformer 不加入 Position Embedding 时满足置换等变性 $Z_{\mathrm{perm}}=PZ$，模型无法感知 token 原本所在的位置。但文本语义依赖词序。因此，该性质会使 Transformer 无法区分不同词序的文本，需要加入 Position Embedding 来提供位置信息。

### (b)
**1.** 能解决，加入 Position Embedding 后 $X_{\mathrm{pos}}=X+\Phi$，若只置换 token 而位置不变，输入变为$X_{\mathrm{perm,pos}}=PX+\Phi$，而原输入整体置换为 $P(X+\Phi)=PX+P\Phi$，因此不同词序会产生不同的输入表示。每个位置具有不同的 Position Embedding，Transformer 可以据此感知 token 的位置和顺序
**2.** 不会。若两个不同位置 $t\ne s$ 的 Position Embedding 相同，则其前两维必须满足$\sin t=\sin s,\cos t=\cos s$，因此 $t=s+2\pi k$，其中 $k$ 为整数。由于 $t,s$ 均为整数，而 $2\pi k$ 仅在 $k=0$ 时为整数，故只能有 $t=s$

## Problem 3

### (a)

这一题实现一个仅含解码器、风格类似 GPT-2 的 Transformer。

**MLP。** MLP 对每个词元独立应用两次线性变换，并在中间使用 GELU：

$$
\operatorname{MLP}(x)=W_2\operatorname{GELU}(W_1x+b_1)+b_2.
$$

第一层将隐藏维度从 $d_{\text{model}}$ 扩展到 $4d_{\text{model}}$，第二层再投影回 $d_{\text{model}}$。

**CausalAttention。** 先从输入得到 $Q$、$K$ 和 $V$，再计算缩放点积注意力：

$$
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.
$$

其中，$M$ 是允许关注分数矩阵下三角部分的因果掩码。位置 $t$ 只能关注位置 $0,\ldots,t$。将未来位置的得分设为 $-\infty$ 后，对应的 softmax 权重为零。多头注意力需要把 $[B,T,d_{\text{model}}]$ 变形为 $[B,h,T,d_h]$，分别计算每个头的注意力，再拼接结果。

**DecoderBlock。** 这里采用 Pre-LN：在注意力和 MLP 之前进行层归一化，然后分别加入对应的残差连接：

$$
\begin{aligned}
H &= X+\operatorname{CausalAttention}(\operatorname{LN}_1(X)),\\
Y &= H+\operatorname{MLP}(\operatorname{LN}_2(H)).
\end{aligned}
$$

**Transformer。** 将词元嵌入与位置嵌入相加，依次通过各个解码器块，最后进行层归一化并通过语言模型输出头，得到 logits：

$$
H^{(0)}=E_{\text{token}}(x)+E_{\text{position}}(0,\ldots,T-1),
$$

$$
\operatorname{logits}=H^{(L)}W_{\text{lm}}^\top.
$$

每一步生成时，选择最后一个位置上 logit 最大的词元：

$$
x_{\text{next}}=\arg\max_v\operatorname{logits}_{-1,v}.
$$

把该词元追加到输入后重复以上步骤。如果序列超过上下文长度，前向计算时只保留最近的 `context_length` 个词元。

### (b)

`get_loss_on_batch` 使用下一词预测。给定输入序列

$$
(x_0,x_1,\ldots,x_{T-1}),
$$

位置 $t$ 的 logits 预测 $x_{t+1}$，因此去掉最后一个位置的 logits 和第一个标签词元：

$$
\operatorname{loss}=
\operatorname{CrossEntropy}\left(
\operatorname{logits}_{[:,0:T-1,:]},
x_{[:,1:T]}
\right).
$$

实现时，先将批次维度与序列维度展平，再传给 `F.cross_entropy`。最后一个位置没有下一词标签，因此不参与损失计算。

训练 100 个批次后，损失整体从约 $10.83$ 下降到约 $10.77$，表明模型已经开始从数据中学习。梯度范数保持在较小范围内，没有明显的梯度爆炸。

![100 个批次的训练损失与梯度范数](/assets/img/posts/cs224n-assignment-3/losses-and-grad-norms.png)

### (c) Bonus

在训练步数固定为 100 的条件下，目标是在相同预算内取得比基线更低的最终损失。我依次进行了三项修改：提高学习率、关闭权重衰减，以及为语言模型输出头加入可学习偏置。

| 设置 | 100 步后的最终损失 |
| --- | ---: |
| 基线 | $\approx 10.77$ |
| 学习率：$10^{-3}$ | $6.449274$ |
| + 关闭权重衰减 | $6.446831$ |
| + 输出头偏置 | **$6.427308$** |

提高学习率带来的收敛改善最大。在这个固定训练预算下，关闭权重衰减略微降低了训练损失；输出头偏置能够直接学习每个词元的整体频率偏移，进一步降低最终损失。
