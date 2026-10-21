---
title: '强化学习 | 策略梯度'
publishDate: 2026-10-21
category: learning
tags: [Reinforcement learning]
language: zh
description: 'Notes on policy gradients: trajectory probabilities, the log-derivative trick, Monte Carlo estimates, and on-policy training.'
heroImage:
  src: /assets/img/posts/multi-armed-bandits/cover.png
  alt: "文章封面"
  color: '#bd7d77'
---

## 策略梯度算法

在一场游戏中，将环境产生的状态 s 和智能体选择的动作 a 按顺序组合起来，就得到一条轨迹：

$$
\tau=(s_1,a_1,s_2,a_2,\ldots,s_T,a_T,s_{T+1})
$$

策略网络的输出是一个动作概率分布，智能体根据这个分布采样，决定实际采取的动作。

给定策略参数 θ，某条轨迹出现的概率为：

$$
\begin{aligned}
p_\theta(\tau)
&= p(s_1)p_\theta(a_1\mid s_1)p(s_2\mid s_1,a_1)
\cdots p_\theta(a_T\mid s_T)p(s_{T+1}\mid s_T,a_T) \\
&= p(s_1)\prod_{t=1}^{T}
p_\theta(a_t\mid s_t)p(s_{t+1}\mid s_t,a_t)
\end{aligned}
$$

其中，$p(s_1)$ 是初始状态概率，$p_\theta(a_t\mid s_t)$ 是智能体选择动作的概率，$p(s_{t+1}\mid s_t,a_t)$ 是环境的状态转移概率。

我们一般无法改变环境的规则，但可以调整 θ，改变动作选择概率，从而改变轨迹出现的概率。将轨迹的累计奖励记为 $R(\tau)$，目标是最大化期望回报：

$$
\bar R_\theta
= \sum_\tau p_\theta(\tau)R(\tau)
= \mathbb E_{\tau\sim p_\theta}[R(\tau)]
$$

为了使用梯度上升更新参数，需要先计算期望回报对 θ 的梯度。对于固定轨迹，$R(\tau)$ 不直接依赖 θ，因此：

$$
\begin{aligned}
\nabla_\theta\bar R_\theta
&= \nabla_\theta\sum_\tau R(\tau)p_\theta(\tau) \\
&= \sum_\tau R(\tau)\nabla_\theta p_\theta(\tau) \\
&= \sum_\tau R(\tau)p_\theta(\tau)
\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} \\
&= \sum_\tau p_\theta(\tau)R(\tau)
\nabla_\theta\log p_\theta(\tau) \\
&= \mathbb E_{\tau\sim p_\theta}
\left[R(\tau)\nabla_\theta\log p_\theta(\tau)\right]
\end{aligned}
$$

这里使用了对数求导公式 $\nabla_\theta\log p_\theta(\tau)=\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}$。整个过程只对轨迹概率求导，因此**奖励函数本身不需要可微**。

接下来，将轨迹概率取对数，把乘积展开为求和：

$$
\log p_\theta(\tau)
= \log p(s_1)
+\sum_{t=1}^{T}\log p_\theta(a_t\mid s_t)
+\sum_{t=1}^{T}\log p(s_{t+1}\mid s_t,a_t)
$$

对于固定轨迹，初始状态概率和环境转移概率与 θ 无关，对 θ 求导后为 0，因此只保留策略项：

$$
\begin{aligned}
\nabla_\theta\log p_\theta(\tau)
&= \nabla_\theta\log p(s_1)
+\sum_{t=1}^{T}\nabla_\theta\log p_\theta(a_t\mid s_t)
+\sum_{t=1}^{T}\nabla_\theta\log p(s_{t+1}\mid s_t,a_t) \\
&= 0+\sum_{t=1}^{T}\nabla_\theta\log p_\theta(a_t\mid s_t)+0 \\
&= \sum_{t=1}^{T}\nabla_\theta\log p_\theta(a_t\mid s_t)
\end{aligned}
$$

因此，计算策略梯度**不需要知道环境的状态转移概率**，只需要计算智能体所选动作的对数概率梯度。

实际中难以遍历所有轨迹，因此使用当前策略采样 N 条轨迹，用样本均值近似期望，再代入上式：

$$
\begin{aligned}
\nabla_\theta\bar R_\theta
&\approx \frac{1}{N}\sum_{n=1}^{N}
R(\tau^n)\nabla_\theta\log p_\theta(\tau^n) \\
&= \frac{1}{N}\sum_{n=1}^{N}
R(\tau^n)\left[
\sum_{t=1}^{T_n}\nabla_\theta\log p_\theta(a_t^n\mid s_t^n)
\right] \\
&= \frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n}
R(\tau^n)\nabla_\theta\log p_\theta(a_t^n\mid s_t^n)
\end{aligned}
$$

其中，n 表示第几条轨迹，t 表示该轨迹中的第几步，$T_n$ 是第 n 条轨迹的长度。

实际计算时，对每一步所选动作的对数概率求梯度，乘以该轨迹的累计奖励；将一条轨迹内的结果相加，再对 N 条轨迹取平均。同一条轨迹中的所有动作都使用相同的累计奖励作为权重。

最后使用估计出的梯度进行梯度上升，α 为学习率：

$$
\theta\leftarrow\theta+
\alpha\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n}
R(\tau^n)\nabla_\theta\log p_\theta(a_t^n\mid s_t^n)
$$
策略网络把状态 $s_t$ 作为输入，输出各动作的概率分布。智能体从中采样动作 $a_t$，与环境交互后，记录每一步的 $(s_t,a_t)$ 和整条轨迹的累计奖励 $R(\tau)$。

实现时可以类比分类问题：把状态看作输入，把实际采样到的动作看作标签，增大该动作的对数概率。但它不是真正的标准答案，因此要用轨迹回报加权：

$$
J(\theta)
= \frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n}
R(\tau^n)\log p_\theta(a_t^n\mid s_t^n)
$$

同一条轨迹中的每一步都乘以该轨迹的 $R(\tau^n)$。回报为正时，更新会提高这些动作的概率；回报为负时，则会降低其概率。对 $J(\theta)$ 求梯度，正好得到前面推导的采样梯度。

在 PyTorch 中通常最小化损失，所以取负号：$L(\theta)=-J(\theta)$。将采样得到的回报作为固定权重，调用自动求导计算 `loss.backward()`，再更新策略参数。

普通策略梯度（on-policy）使用当前策略采样。参数更新后，应重新采样，再进行下一次更新；旧数据不能直接当作新策略的数据反复使用。
