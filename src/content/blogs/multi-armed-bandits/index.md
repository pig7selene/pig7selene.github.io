---
title: "强化学习 | 多臂老虎机"
publishDate: 2026-10-12
category: learning
tags: [Reinforcement learning]
language: zh
description: 'An introduction to multi-armed bandits, regret, and three approaches to balancing exploration and exploitation: ETC, UCB, and Thompson sampling.'
heroImage:
  src: /assets/img/posts/multi-armed-bandits/cover.png
  alt: "文章封面"
  color: '#bd7d77'
---

<aside class="download-callout" role="note">
  <p>You can download the <a href="/output/pdf/multi-armed-bandits.tex">LaTeX</a> and <a href="/output/pdf/multi-armed-bandits.pdf">PDF</a>.</p>
</aside>

## 多臂老虎机

先介绍一个非常经典的问题，多臂老虎机（MAB），简单来说就是一个 $n$ 臂老虎机，随机拉一个臂会给 $[0,1]$ 的奖励值，假设每个时刻我们只能拉下一个拉杆，我们的目标就是让期望最大化，对于每个动作我们可以定义其期望为$Q(a)=\mathbb{E}_{r\sim R(\cdot\mid a)}[r]$ ，肯定存在至少一个拉杆，期望不小于其他拉杆，期望为 $Q^* = max_{a \in A} Q(a)$，我们引入一个懊悔（regret）概念，定义为当前选中的和最优的差距，形式化来说就是 $R(a) = Q^* - Q(a) \ge 0$，那么我们要做的就是最小化累计懊悔，接下来会介绍三个比较经典的算法

### 2.1 先探索后确定（ETC）

如题，探索和确定就是这个算法的关键，探索指我们对MAB的不同拉杆进行多次的尝试，最后对于每个臂我们可以得到一个均值，接下来就是确定均值（奖励）最高的拉杆，然后一直拉这个，其 regret 为：

$$
R(T)=L\sum_{i=1}^{n}\Delta_i+(T-nL)\sum_{i=2}^{n}\Delta_i\Pr\left(\hat{\mu}_i>\max_{j\ne i}\hat{\mu}_j\right)
$$

因为臂 1 最优，有 $\Pr(\hat{\mu}_i>\max_{j\ne i}\hat{\mu}_j)\le \Pr(\hat{\mu}_i>\hat{\mu}_1)$。由 Hoeffding 不等式，

$$
\Pr(\hat{\mu}_i>\hat{\mu}_1)\le \exp\left(-\frac{L\Delta_i^2}{2}\right)
$$

因此

$$
R(T)\le \sum_{i=1}^{n}\left(L\Delta_i+T\Delta_i\exp\left(-\frac{L\Delta_i^2}{2}\right)\right)
$$

对 $\Delta_i$ 取最坏情况，可得 $R(T)\le n\left(L+\frac{Te^{-1/2}}{\sqrt L}\right)$。令 $L=\Theta(T^{2/3})$，最终有 $R(T)=\Theta(nT^{2/3})$。

ETC实现了一个次线性懊悔，但是显然并不是最好，有几个缺点

- 从头到尾都是选择一个臂，但是如果一开始我们由于随机波动选择了一个并非最优的拉杆，会产生很大的累积懊悔

- 探索的步数不好确定，容易太小或者太大

- 无论已经获得的奖励如何，每个臂都固定被尝试 $L$ 次，效率太低

### 2.2 上置信界（UCB）

上置信界算法是一种基于不确定性的算法，举个例子，比如我们现在有一个 $n$ 臂老虎机，其中 $n - 1$ 个臂都被拉过了许多次，有了大概的奖励分布，而剩下一个只拉了1次，奖励是 0，有些人可能比较保守会选择 $n - 1$ 个中最高的，但是我相信肯定有赌狗会选择剩下那一个(x)，因此UCB算法就增加了一个 bonus ，假设我们在第 $t$ 次拉动前，第 $i$ 个臂已被拉动了 $T_i(t-1)$ 次，其前几次的经验均值为 $\hat{\mu}_{i,T_i(t-1)}$，并且我们保证前 $n$ 轮会把每个臂拉一遍，对于 $t\ge n+1$，算法在第 $t$ 次会选择最大化以下值的臂：

$$
U_{i,t}=
\hat{\mu}_{i,T_i(t-1)}+
\sqrt{\frac{2\ln t}{T_i(t-1)}}
$$

如果取到最大值的有多个臂，则随机选择其中一个。这个算法就是对于不确定性的事件给更多的bonus，否则剩下的这个臂可能永远不会被考虑到但是可能它才是最优，这个 bonus 可以由 Hoeffding 不等式得到。对于取值在 $[0,1]$ 的独立随机变量，其真实期望与经验均值之间有

$$
\mathbb P\left(\mu_i\ge \hat\mu_i+u\right)\le e^{-2T_i u^2}
$$

也就是说，$\hat\mu_i+u$ 可以看作真实期望 $\mu_i$ 的一个上置信界。令右侧失败概率为 $p$，则 $u=\sqrt{\frac{-\ln p}{2T_i}}$，UCB 中通常令 $p=t^{-4}$，于是得到 $u=\sqrt{\frac{2\ln t}{T_i}}$

正好就是上面的 bonus。因此，$T_i$ 越小，说明该臂探索得越少、不确定性越大，bonus 就越大；随着该臂被不断拉动，$T_i$ 增大，bonus 会逐渐减小

### 2.3 汤普森采样（Thompson Sampling）

汤普森采样的思路是，我们不知道每个臂的真实期望，那就用一个概率分布来描述它，每轮从各个分布中采样一个可能的期望，然后选择采样值最大的臂。

先考虑奖励只有 $0$ 和 $1$ 的情况，假设第 $i$ 个臂的奖励服从伯努利分布，未知的期望为 $\mu_i$，我们给它设置一个 Beta 先验：

$$
\mu_i\sim\operatorname{Beta}(\alpha_i,\beta_i)
$$

一般初始化为 $\alpha_i=\beta_i=1$，即 $[0,1]$ 上的均匀分布。在第 $t$ 轮，从每个臂当前的后验中采样，然后选择最大的：

$$
\theta_{i,t}\sim\operatorname{Beta}(\alpha_i,\beta_i),
\qquad
a_t=\arg\max_i\theta_{i,t}
$$

观察实际奖励 $r_t$ 后，更新被选中臂的参数：

$$
\alpha_{a_t}\leftarrow\alpha_{a_t}+r_t,\qquad
\beta_{a_t}\leftarrow\beta_{a_t}+1-r_t
$$

也就是成功时 $\alpha$ 加一，失败时 $\beta$ 加一，其他臂不变。这个更新可以由贝叶斯公式得到。

## 总结

这三个算法都在解决探索和利用之间的平衡问题：ETC 先探索再固定选择，UCB 通过 bonus 鼓励尝试不确定性大的臂，汤普森采样则通过后验采样实现探索。它们的目标都是根据有限的反馈逐渐找到最优臂，并尽可能减少这个过程中的累计懊悔。因此解决探索和利用之间的平衡问题在强化学习的学习过程中非常重要，是强化学习试错法中的必备技术
