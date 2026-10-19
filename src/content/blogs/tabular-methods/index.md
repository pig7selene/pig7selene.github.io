---
title: "强化学习 | 表格型方法"
publishDate: 2026-10-17
category: learning
tags: [Reinforcement learning]
language: zh
description: 'Model-free prediction and control with Q-tables, Monte Carlo estimation, temporal-difference learning, exploring starts, and epsilon-greedy policies.'
heroImage:
  src: /assets/img/posts/multi-armed-bandits/cover.png
  alt: "文章封面"
  color: '#bd7d77'
---

## 有模型与免模型方法

- 有模型：已知状态转移规律和奖励模型，动态规划是一个典型例子

- 免模型：状态转移规律和奖励模型是未知的，让智能体和环境交互，采集数据改进策略，从而获得更多的的奖励

## Q 表格

Q 表格就是一个记录状态-动作价值 $Q(s,a)$ 的表。$Q(s,a)$ 记录的不是单步奖励，而是在当前状态执行某个动作、随后遵循策略时的期望回报。使用一步时序差分方法时，每走一步就更新 $Q$ 表格，不必等到回合结束。

## 免模型预测

### 蒙特卡洛策略评估

蒙特卡洛（Monte Carlo，MC）方法通过与环境交互，采样完整回合，用实际回报的平均值估计给定策略 π 的状态价值

状态价值 $V_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]$ 表示：从状态 s 出发，按照策略 π 行动时，能够获得的期望回报

因此，可以反复执行策略 π，记录每次访问状态 s 之后的实际回报，再取平均值。对于终止时刻为 T 的回合，回报为：

$G_t=r_{t+1}+\gamma r_{t+2}+\cdots+\gamma^{T-t-1}r_T$

每个回合结束后，计算所访问状态对应的回报。对状态 s，维护访问次数 N(s) 和累计回报 S(s)：

- $N(s)\leftarrow N(s)+1$；
- $S(s)\leftarrow S(s)+G_t$；
- $V(s)=S(s)/N(s)$

在策略固定、相关状态得到充分采样等条件下，随着样本数量增加，估计值会趋近真实价值 $V_\pi(s)$

利用增量均值公式 $\mu_n=\mu_{n-1}+\frac{1}{n}(x_n-\mu_{n-1})$，可以直接更新价值，无须保存累计回报：

$$
V(s)\leftarrow V(s)+\frac{1}{N(s)}[G_t-V(s)]
$$

可以将步长 $1/N(s)$ 替换为学习率 α：$V(s)\leftarrow V(s)+\alpha[G_t-V(s)]$

其中，$G_t-V(s)$ 是实际回报与当前估计之间的误差。每次更新都让 V(s) 向实际回报靠近一些。步长为 $1/N(s)$ 时得到算术平均；固定 α 时则更重视近期样本

### 时序差分

一步时序差分TD(0)是最简单的算法，每往前走一步，就做一步自举，回报满足 $G_t=r_{t+1}+\gamma G_{t+1}$。虽然还不知道后续的实际回报 $G_{t+1}$，但可以用当前估计的 $V(s_{t+1})$ 代替它，因此每走一步就能更新：

$$
V(s_t)\leftarrow V(s_t)+\alpha[r_{t+1}+\gamma V(s_{t+1})-V(s_t)]
$$

$r_{t+1}+\gamma V(s_{t+1})$是时序差分目标，由实际获得的即时奖励和后继状态的估计价值组成。$\delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t)$是时序差分误差，表示更新目标与当前估计之间的差距

更新也可以简写为 $V(s_t)\leftarrow V(s_t)+\alpha\delta_t$，即让当前估计向时序差分目标靠近一些，若下一状态是终止状态，则其价值取 0

我们可以把一步时序差分改变步数变成 **n 步时序差分**，先观察 n 步，再用后续状态的估计价值补上剩余回报：

$$
G_t^{(n)}=r_{t+1}+\gamma r_{t+2}+\cdots+\gamma^{n-1}r_{t+n}+\gamma^nV(s_{t+n})
$$

更新为

$$
V(s_t)\leftarrow V(s_t)+\alpha[G_t^{(n)}-V(s_t)]
$$

$n$ 越大，使用的实际奖励越多，在回合制任务中，当回报覆盖直到终止的所有剩余步骤时，就得到蒙特卡洛更新

### 蒙特卡洛与时序差分方法的比较

- 时序差分可以在线更新，效率高；而蒙特卡洛得要到回合结束才更新
- 时序差分可以从不完整序列上学习；而蒙特卡洛方法不行
- 时序差分可以在没有终止状态的持续任务中学习；而蒙特卡洛只能在有终止的环境下学习
- 时序差分利用马尔可夫性质，用当前状态的价值概括后续回报；蒙特卡洛方法则直接使用完整轨迹的实际回报

## 免模型控制

在不知道环境模型，即不知道奖励函数 $R(s,a)$ 和状态转移概率 $P(s'\mid s,a)$ 时，无法直接计算 $Q$ 函数，因此可以使用蒙特卡洛法，通过与环境交互得到的完整回合来估计动作价值函数 $Q_\pi(s,a)$。

广义策略迭代主要包含两个不断交替的过程：先固定当前策略 $\pi$，通过蒙特卡洛采样估计 $Q_\pi$，再根据当前 $Q$ 函数进行贪心更新

$$
\pi'(s)\in\arg\max_a Q_\pi(s,a)
$$

不断重复：

$$
\pi \rightarrow Q_\pi \rightarrow \pi' \rightarrow Q_{\pi'} \rightarrow \cdots
$$

在满足适当的探索与评估条件下，策略和价值函数逐渐逼近最优策略 $\pi^*$ 和最优动作价值函数 $Q^*$。

为了使用蒙特方法估计动作价值函数 Q，需要保证所有状态-动作对 $(s,a)$ 都能够被充分采样。并且回合要有探索性开始，也就是每个回合从随机的状态-动作对 $(s_0,a_0)$ 开始，并保证任意 $(s,a)$ 都有非零概率被选为起点，从而使所有状态-动作对最终都能被访问

<img src="/assets/img/posts/tabular-methods/monte-carlo-exploring-starts.png" alt="采用探索性开始的蒙特卡洛控制伪代码：从随机状态—动作对采样回合，平均回报并贪心地改进策略。" style="display: block; width: min(100%, 760px); height: auto; margin: 1.5rem auto 0.5rem;" />

在首次访问版本中，只在状态—动作对**此前未在该回合出现过**时更新它。这是上图伪代码中首次访问条件应表达的含义。

为了兼顾探索和利用，采用 ε-贪心策略：以 $1-\varepsilon$ 的概率选择 Q 值最大的动作，以 ε 的概率从所有动作中均匀随机选择。训练初期多探索，随后逐渐减小 ε，更多地利用已有经验，蒙特卡洛控制不断重复：使用 ε-贪心策略采样完整回合，根据实际回报更新 Q 表格，再根据 Q 表格改进策略。固定相同的 ε，在准确评估 $Q_\pi$ 的前提下，将原 ε-贪心策略更新为关于 $Q_\pi$ 的 ε-贪心策略 π′，可保证 $V_{\pi'}(s)\ge V_\pi(s)$

<img src="/assets/img/posts/tabular-methods/monte-carlo-epsilon-greedy.png" alt="采用 ε-贪心探索的蒙特卡洛控制伪代码：采样回合，使用增量均值更新动作价值，并改进策略。" style="display: block; width: min(100%, 760px); height: auto; margin: 1.5rem auto 0.5rem;" />

### Sarsa算法

Sarsa 把原本时序差分的 $V$ 换成了 $Q$ 
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)\right]
$$
Sarsa的更新公式如下
$$
Q(S,A) \leftarrow Q(S,A) + \alpha \left(R + \gamma Q(S',A') - Q(S,A)\right)
$$
同理，我们也可以得到 $n$ 步Sarsa的回报函数和更新过程
$$
Q_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}Q_t^n \\
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left(Q_t^\lambda - Q(s_t, a_t)\right)
$$

### Q-learning 算法

Q学习算法有两种策略，行为策略和目标策略，目标策略 $\pi$ 直接在Q表格上做贪心
$$
\pi(s_{t+1}) = \arg\max_{a'} Q(s_{t+1}, a')
$$
行为策略 $\mu$ 我们采用 $\epsilon$-贪心策略，我们可以构造 Q学习 目标，Q学习的下一个动作都是通过 arg max 操作选出来的
$$
\begin{aligned}
r_{t+1} + \gamma Q(s_{t+1}, A')
&= r_{t+1} + \gamma Q\left(s_{t+1}, \arg\max_{a'} Q(s_{t+1}, a')\right) \\
&= r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')
\end{aligned}
$$
接着我们就可以写出更新过程了
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t)\right]
$$


### Sarsa和Q学习的区别

这两个算法一个是同策略算法，一个是异策略算法， 所以最大区别就是Sarsa只用一个策略学习和交互，而Q学习有两套策略，目标策略用来制定一个相对保守的计划，但是目标策略并不会与环境交互，行为策略用来和环境交互，把数据再喂给目标策略

Sarsa 和 Q-learning 更新过程的区别在于更新 Q 值的目标，Sarsa 使用实际选择的下一步动作，需要 $(S,A,R,S',A')$，因此会考虑探索风险，策略相对保守；Q-learning 直接使用下一状态的最大 Q 值，只需要 $(S,A,R,S')$，不考虑探索动作对更新目标的影响，因此相对激进
