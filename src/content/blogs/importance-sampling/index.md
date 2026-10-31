---
title: '强化学习 | 重要性采样'
publishDate: 2026-10-31
category: learning
tags: [Reinforcement learning]
language: zh
description: 'Notes on importance sampling, unbiased estimation, variance, and reusing policy data with likelihood ratios.'
heroImage:
  src: /assets/img/posts/multi-armed-bandits/cover.png
  alt: "文章封面"
  color: '#bd7d77'
---

## 重要性采样（Importance Sampling）

设目标分布为 $p$，采样分布为 $q$，要求 $p(x)>0\Rightarrow q(x)>0$。定义权重 $w(x)=p(x)/q(x)$，从 $q$ 独立采样 $x_1,\dots,x_N$，则：

$$
\begin{aligned}
\mu &= \mathbb{E}_{x\sim p}[f(x)]=\int f(x)p(x)\,dx \\
&= \int f(x)\frac{p(x)}{q(x)}q(x)\,dx
= \mathbb{E}_{x\sim q}[f(x)w(x)] \\
\hat{\mu} &= \frac{1}{N}\sum_{i=1}^{N}f(x_i)w(x_i) \\
\mathbb{E}[\hat{\mu}] &= \frac{1}{N}\sum_{i=1}^{N}\mathbb{E}_{x_i\sim q}[f(x_i)w(x_i)]=\mu
\end{aligned}
$$

两种采样方式的期望相同，但方差不一定相同：

$$
\begin{aligned}
\operatorname{Var}_{x\sim p}[f(x)]
&= \mathbb{E}_{x\sim p}[f(x)^2]-\mu^2 \\
\operatorname{Var}_{x\sim q}[f(x)w(x)]
&= \mathbb{E}_{x\sim q}[f(x)^2w(x)^2]-\mu^2 \\
&= \int f(x)^2\frac{p(x)^2}{q(x)}\,dx-\mu^2 \\
&= \mathbb{E}_{x\sim p}\left[f(x)^2\frac{p(x)}{q(x)}\right]-\mu^2
\end{aligned}
$$

$$
\begin{gathered}
\operatorname{Var}_{x\sim q}[f(x)w(x)]-\operatorname{Var}_{x\sim p}[f(x)]
= \mathbb{E}_{x\sim p}\left[f(x)^2\left(\frac{p(x)}{q(x)}-1\right)\right] \\[6pt]
\operatorname{Var}[\hat{\mu}]
= \frac{1}{N}\operatorname{Var}_{x\sim q}[f(x)w(x)]
\end{gathered}
$$

上述方差推导要求相关二阶矩有限。当部分区域的 $p(x)/q(x)$ 极大时，加权后的方差可能很大，使有限样本下的估计不稳定。

当 $p(x)$ 主要集中在左侧、$q(x)$ 主要集中在右侧，且左侧 $f(x)<0$、右侧 $f(x)>0$ 时，假设真实期望 $\mu<0$。从 $q$ 采样时，少量样本可能全部落在右侧，使有限样本估计为正：

$$
\begin{aligned}
\mu &= \mathbb{E}_{x\sim p}[f(x)]
= \mathbb{E}_{x\sim q}\left[f(x)\frac{p(x)}{q(x)}\right]<0 \\
\hat{\mu}_N &= \frac{1}{N}\sum_{i=1}^{N}f(x_i)\frac{p(x_i)}{q(x_i)}>0
\end{aligned}
$$

左侧虽然很难被 $q$ 采到，但那里 $p(x)$ 大、$q(x)$ 小，因此权重 $p(x)/q(x)$ 极大。一旦采到左侧样本，就可能产生巨大的负贡献，使估计结果剧烈变化。

**理论期望始终相等，但有限样本估计可能相差很大，甚至符号相反。** 在满足可积条件时，随着样本数增加，估计会收敛到真实期望；但当重要性权重波动很大时，可能需要大量样本才能得到稳定结果。

设目标策略为 $\pi_\theta$，采样策略为固定的 $\pi_{\theta'}$。通过重要性采样，可以使用 $\theta'$ 采集的数据多次更新 $\theta$：

$$
\begin{aligned}
\nabla_\theta J(\theta)
&= \mathbb{E}_{\tau\sim p_\theta}\left[R(\tau)\nabla_\theta\log p_\theta(\tau)\right] \\
&= \mathbb{E}_{\tau\sim p_{\theta'}}\left[\frac{p_\theta(\tau)}{p_{\theta'}(\tau)}R(\tau)\nabla_\theta\log p_\theta(\tau)\right]
\end{aligned}
$$

实际按状态—动作计算梯度，使用优势函数 $A^\theta(s_t,a_t)=Q^\theta(s_t,a_t)-V^\theta(s_t)$。重要性权重可以拆成状态分布比与动作概率比：

$$
\begin{aligned}
p_\theta(s_t,a_t) &= p_\theta(s_t)\pi_\theta(a_t\mid s_t) \\
\frac{p_\theta(s_t,a_t)}{p_{\theta'}(s_t,a_t)}
&= \frac{p_\theta(s_t)}{p_{\theta'}(s_t)}
\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta'}(a_t\mid s_t)}
\end{aligned}
$$

当新旧策略足够接近时，近似忽略状态分布差异，并用旧策略数据估计的优势 $A^{\theta'}$ 替代 $A^\theta$。记动作概率比为 $r_t(\theta)$：

$$
\begin{gathered}
p_\theta(s_t)\approx p_{\theta'}(s_t),\qquad A^\theta(s_t,a_t)\approx A^{\theta'}(s_t,a_t) \\
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta'}(a_t\mid s_t)}
\end{gathered}
$$

由此得到近似梯度形式（省略时间步求和及权重）：

$$
\nabla_\theta J(\theta)\approx
\mathbb{E}_{(s_t,a_t)\sim p_{\theta'}}\left[
r_t(\theta)A^{\theta'}(s_t,a_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right]
$$

更新时，$\theta'$ 与估计好的优势保持固定，只优化 $\theta$。优势为正时提高动作概率，为负时降低动作概率。**状态分布与优势的替换属于近似，新旧策略差异过大时，近似误差和重要性权重的方差都可能增大。**
