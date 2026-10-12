---
title: '强化学习 | 多臂老虎机'
publishDate: 2026-10-12
category: learning
tags: [Reinforcement learning]
language: en
description: 'An introduction to multi-armed bandits, regret, and three approaches to balancing exploration and exploitation: ETC, UCB, and Thompson sampling.'
heroImage:
  src: /assets/img/posts/multi-armed-bandits/cover.png
  alt: 'A traveler looking over a desert city and lake at sunset'
  color: '#bd7d77'
---

> You can download the [LaTeX](/output/pdf/multi-armed-bandits.tex) and [PDF](/output/pdf/multi-armed-bandits.pdf), and you can also read the [Chinese version](https://notes.sjtu.edu.cn/s/NeAXHWf9V) here.

## Multi-Armed Bandits

Let us begin with a classic problem: the multi-armed bandit (MAB). Imagine a machine with $n$ arms. Pulling an arm produces a random reward in $[0,1]$, and at each time step we can pull only one arm. Our goal is to maximize the expected reward. For each action $a$, define its expected reward as $Q(a)=\mathbb{E}_{r\sim R(\cdot\mid a)}[r]$. At least one arm must have an expected reward no smaller than that of any other arm, so let $Q^*=\max_{a\in A}Q(a)$.

We now introduce **regret**, the gap between the optimal expected reward and that of the arm we choose:

$$
R(a)=Q^*-Q(a)\ge 0.
$$

Our aim is to minimize cumulative regret. The following sections introduce three classic algorithms for doing so.

### 2.1 Explore-Then-Commit (ETC)

As its name suggests, this algorithm has two stages: exploration and commitment. During exploration, we try each arm several times and compute its empirical mean reward. We then commit to the arm with the highest empirical mean and pull it for the rest of the horizon. Its regret is

$$
R(T)=L\sum_{i=1}^{n}\Delta_i+(T-nL)\sum_{i=2}^{n}\Delta_i\Pr\left(\hat{\mu}_i>\max_{j\ne i}\hat{\mu}_j\right).
$$

Because arm $1$ is optimal, $\Pr(\hat{\mu}_i>\max_{j\ne i}\hat{\mu}_j)\le\Pr(\hat{\mu}_i>\hat{\mu}_1)$. By Hoeffding's inequality,

$$
\Pr(\hat{\mu}_i>\hat{\mu}_1)\le\exp\left(-\frac{L\Delta_i^2}{2}\right).
$$

Therefore,

$$
R(T)\le\sum_{i=1}^{n}\left(L\Delta_i+T\Delta_i\exp\left(-\frac{L\Delta_i^2}{2}\right)\right).
$$

Taking the worst case over $\Delta_i$ gives $R(T)\le n\left(L+\frac{Te^{-1/2}}{\sqrt L}\right)$. Choosing $L=\Theta(T^{2/3})$ then yields $R(T)=\Theta(nT^{2/3})$.

ETC achieves sublinear regret, but it has some clear shortcomings:

- After exploration, it sticks with a single arm. If random fluctuations cause us to commit to a suboptimal arm, the resulting cumulative regret can be large.
- The number of exploration rounds is difficult to choose: it is easy to explore either too little or too much.
- Every arm is tried a fixed $L$ times regardless of the rewards already observed, which can be inefficient.

### 2.2 Upper Confidence Bound (UCB)

UCB is an algorithm based on uncertainty. Suppose that, in an $n$-armed bandit, we have pulled $n-1$ arms many times and have a rough idea of their reward distributions. The remaining arm has been pulled only once and returned a reward of $0$. A cautious person might pick the best of the well-tested arms, but a more adventurous person might still try the last one. UCB formalizes that instinct by adding a **bonus** for uncertainty.

Before the pull at time $t$, let $T_i(t-1)$ be the number of times arm $i$ has been pulled, and let $\hat{\mu}_{i,T_i(t-1)}$ be its empirical mean reward. We first pull each arm once during the first $n$ rounds. For $t\ge n+1$, the algorithm chooses an arm that maximizes

$$
U_{i,t}=\hat{\mu}_{i,T_i(t-1)}+\sqrt{\frac{2\ln t}{T_i(t-1)}}.
$$

If several arms attain the maximum, we choose randomly among them. The bonus gives more weight to arms about which we are less certain. Otherwise, an arm that has been sampled only a few times might never be reconsidered, even though it could be optimal.

The bonus follows from Hoeffding's inequality. For independent random variables taking values in $[0,1]$, the true mean and empirical mean satisfy

$$
\mathbb{P}\left(\mu_i\ge\hat\mu_i+u\right)\le e^{-2T_i u^2}.
$$

Thus, $\hat\mu_i+u$ can be regarded as an upper confidence bound for the true mean $\mu_i$. If we set the failure probability on the right to $p$, then $u=\sqrt{\frac{-\ln p}{2T_i}}$. UCB commonly takes $p=t^{-4}$, giving $u=\sqrt{\frac{2\ln t}{T_i}}$—exactly the bonus above.

When $T_i$ is small, the arm has been explored less, its uncertainty is greater, and its bonus is larger. As we pull that arm more often, $T_i$ grows and the bonus gradually shrinks.

### 2.3 Thompson Sampling

The idea behind Thompson sampling is to represent our uncertainty about each arm's true expected reward with a probability distribution. In each round, we sample one possible mean from every arm's distribution and choose the arm with the largest sampled value.

First consider rewards that can only be $0$ or $1$. Suppose the reward from arm $i$ follows a Bernoulli distribution with unknown mean $\mu_i$. We assign it a Beta prior:

$$
\mu_i\sim\operatorname{Beta}(\alpha_i,\beta_i).
$$

We usually initialize $\alpha_i=\beta_i=1$, which gives a uniform distribution on $[0,1]$. In round $t$, we sample from each arm's current posterior and select the largest sample:

$$
\theta_{i,t}\sim\operatorname{Beta}(\alpha_i,\beta_i),
\qquad
a_t=\arg\max_i\theta_{i,t}.
$$

After observing the actual reward $r_t$, we update the parameters of the chosen arm:

$$
\alpha_{a_t}\leftarrow\alpha_{a_t}+r_t,
\qquad
\beta_{a_t}\leftarrow\beta_{a_t}+1-r_t.
$$

In other words, a success increments $\alpha$ and a failure increments $\beta$; the other arms remain unchanged. This update follows from Bayes' rule.

## Summary

All three algorithms address the balance between exploration and exploitation. ETC explores first and then commits to a fixed choice. UCB uses a bonus to encourage trying arms with greater uncertainty. Thompson sampling explores by sampling from posterior distributions. Each aims to identify the best arm from limited feedback while keeping cumulative regret as small as possible. Finding this balance is therefore an important part of learning reinforcement learning and an essential technique for trial-and-error methods.
