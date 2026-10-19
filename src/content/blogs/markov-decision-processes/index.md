---
title: "强化学习 | 马尔可夫决策过程"
publishDate: 2026-10-13
category: learning
tags: [Reinforcement learning]
language: zh
description: 'Notes on Markov processes, Bellman expectation and optimality equations, optimal policies, policy iteration, and value iteration.'
heroImage:
  src: /assets/img/posts/multi-armed-bandits/cover.png
  alt: "文章封面"
  color: '#bd7d77'
---

<aside class="download-callout" role="note">
  <p>You can download the <a href="/output/pdf/markov-decision-processes.tex">LaTeX</a> and <a href="/output/pdf/markov-decision-processes.pdf">PDF</a>.</p>
</aside>

## 马尔可夫过程

**马尔可夫性质**：未来状态只和当前状态有关，和过去的无关

我们可以用状态转移矩阵 $P$ 来描述状态转移 $p(s_{t+1}=s' \mid s_t=s)$：

$$
P=
\begin{pmatrix}
p(s_1\mid s_1) & p(s_2\mid s_1) & \cdots & p(s_N\mid s_1) \\
p(s_1\mid s_2) & p(s_2\mid s_2) & \cdots & p(s_N\mid s_2) \\
\vdots & \vdots & \ddots & \vdots \\
p(s_1\mid s_N) & p(s_2\mid s_N) & \cdots & p(s_N\mid s_N)
\end{pmatrix}
$$

## 马尔可夫奖励过程

在马尔可夫链的基础上增加奖励函数，奖励函数给出即时奖励的期望。对于有限回合，回报定义为奖励的折扣叠加。假设时刻 $t$ 后的奖励序列为 $r_{t+1},r_{t+2},r_{t+3},\cdots$，那么回报为

$$
r_{t+1} + \gamma r_{t+2} + \gamma ^ 2r_{t+3}+\cdots + \gamma ^ {T-t-1}r_T
$$

$T$ 是最后时刻，$\gamma$ 是折扣因子，显然越往后折扣越多，说明我们更想要现在的奖励，我们定义状态价值函数为回报的期望，即

$$
\begin{aligned}
V^t(s)
&=\mathbb{E}[G_t\mid s_t=s] \\
&=\mathbb{E}\left[r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\cdots+\gamma^{T-t-1}r_T\mid s_t=s\right]
\end{aligned}
$$

### Bellman 方程

**贝尔曼方程**把状态价值拆成两部分：**当前一步的期望奖励**，加上**下一状态价值的折扣期望**。

$$
V(s)=R(s)+\gamma\sum_{s'}p(s'\mid s)V(s')
$$

其中 $p(s'\mid s)$ 是从 $s$ 转移到 $s'$ 的概率，$\gamma\sum_{s'}p(s'\mid s)V(s')$ 可以看作未来的折扣总和

我们可以把贝尔曼方程写成矩阵的形式

$$
\begin{pmatrix}
V(s_1) \\
V(s_2) \\
\vdots \\
V(s_N)
\end{pmatrix}
=
\begin{pmatrix}
R(s_1) \\
R(s_2) \\
\vdots \\
R(s_N)
\end{pmatrix}
+
\gamma
\begin{pmatrix}
p(s_1\mid s_1) & p(s_2\mid s_1) & \cdots & p(s_N\mid s_1) \\
p(s_1\mid s_2) & p(s_2\mid s_2) & \cdots & p(s_N\mid s_2) \\
\vdots & \vdots & \ddots & \vdots \\
p(s_1\mid s_N) & p(s_2\mid s_N) & \cdots & p(s_N\mid s_N)
\end{pmatrix}
\begin{pmatrix}
V(s_1) \\
V(s_2) \\
\vdots \\
V(s_N)
\end{pmatrix}
$$

把所有状态的 Bellman 方程合起来，可写成矩阵形式 $V=R+\gamma PV$，移项后得到**解析解**

$$
V=(I-\gamma P)^{-1}R
$$

### 状态价值的计算

- 蒙特卡洛：从状态 $s$ 出发采样多条完整轨迹，计算各轨迹的回报并取平均
- 动态规划：已知奖励和转移概率时，反复迭代 Bellman 方程更新价值，直到收敛，用后继状态的当前价值估计更新前一状态，这称为自举（bootstrapping）。

## 马尔可夫决策过程

### MDP 与 MRP

MRP是直接在 $S$ 中选择下一个的状态，而 MDP 先经过动作集，从动作 $a$ 再转移，所以 MDP 的下一步动作是由智能体来决定的

<img src="/assets/img/posts/markov-decision-processes/mrp-mdp-transitions.png" alt="MRP 直接从状态转移到后继状态；MDP 先根据策略选择动作，再转移到后继状态。" style="display: block; width: min(100%, 680px); height: auto; margin: 1.5rem auto;" />

### Bellman 期望方程

固定策略 $\pi$ 后，$V_\pi(s)$ 是从状态 $s$ 出发遵循 $\pi$ 的期望回报；$Q_\pi(s,a)$ 是在 $s$ 先执行 $a$、之后遵循 $\pi$ 的期望回报。两者满足

$$
\begin{aligned}
V_\pi(s)&=\sum_a\pi(a\mid s)Q_\pi(s,a),\\
Q_\pi(s,a)&=R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V_\pi(s').
\end{aligned}
$$

将第一个式子代入第二个式子，可得到只含 $Q_\pi$ 的 Bellman 期望方程

$$
Q_\pi(s,a)=R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)
\sum_{a'}\pi(a'\mid s')Q_\pi(s',a').
$$

同理可得只含 $V_\pi$ 的 Bellman 期望方程

$$
V_\pi(s)=\sum_a\pi(a\mid s)\left[R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V_\pi(s')\right].
$$

## 最优策略

### 最优价值函数

最佳价值函数定义为 $V^*(s)=\max_{\pi}V_{\pi}(s)$。达到最佳价值的策略称为最佳策略。当取得最佳动作价值函数后，我们可以在每个状态选择一个使 $Q^*(s,a)$ 最大的动作 $a^*(s)$，得到确定性的最佳策略：

$$
\pi^*(a\mid s)=
\begin{cases}
1, & a=a^*(s) \\
0, & \text{otherwise}
\end{cases}
$$

这里介绍两种常用的搜索最佳策略的方法，策略迭代和价值迭代

### 策略迭代

策略迭代包含两个重要的步骤，策略评估和策略提升

策略评估用来计算一个策略的状态价值函数，然后据此改进这个策略，也就是策略提升，因此策略迭代的过程就是不断进行策略评估和策略提升，不断迭代直到收敛到最佳，接下来细讲一下策略评估和策略提升是怎么做的

策略评估：固定策略 π，反复用 Bellman 期望方程更新状态价值：

$$
V_{k+1}(s)=\sum_a\pi(a\mid s)
\left[R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V_k(s')\right].
$$

策略提升：根据评估得到的价值，在每个状态选择预期回报最高的动作：

$$
\pi'(s)=\arg\max_a
\left[R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V_\pi(s')\right].
$$

### Bellman 最优方程

最优状态价值与最优动作价值满足：

$$
V^*(s)=\max_a Q^*(s,a),
$$

$$
Q^*(s,a)=R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V^*(s').
$$

将两式结合，得到两种常见形式：

$$
V^*(s)=\max_a\left[R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V^*(s')\right],
$$

$$
Q^*(s,a)=R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)\max_{a'}Q^*(s',a').
$$

求出最优价值后，在每个状态选择使 Q* 最大的动作，就得到最优策略：

$$
\pi^*(s)\in\arg\max_a Q^*(s,a).
$$

### 价值迭代

价值迭代则在每轮只更新一次各状态的价值，并在更新时直接选择价值最大的动作

$$
V_{k+1}(s)=\max_a\left[R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V_k(s')\right].
$$

从任意初始价值 V₀ 开始，对所有状态反复应用这个更新式，直到前后两轮的价值变化足够小。此时 V 接近最优状态价值 V*，再据此选出策略：

$$
\pi^*(s)\in\arg\max_a
\left[R(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V^*(s')\right].
$$

### 策略迭代与价值迭代的比较

- **策略迭代（Policy Iteration）**：固定当前策略，反复评估其价值，再根据评估结果提升策略；交替进行，直到策略不再改变。
- **价值迭代（Value Iteration）**：每轮只更新一次各状态的价值，并直接选择当前估计下价值最大的动作；反复更新，直到价值收敛。

策略迭代更稳健，评估充分，价值迭代有点像贪心，具体怎么选得看情况
