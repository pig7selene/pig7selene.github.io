---
title: 'Reinforcement Learning | Tabular Methods'
publishDate: 2026-10-17
category: learning
tags: [Reinforcement learning]
language: en
description: 'Model-free prediction and control with Q-tables, Monte Carlo estimation, temporal-difference learning, exploring starts, and epsilon-greedy policies.'
heroImage:
  src: /assets/img/posts/multi-armed-bandits/cover.png
  alt: 'A traveler looking over a desert city and lake at sunset'
  color: '#bd7d77'
---

## Model-Based and Model-Free Methods

- **Model-based methods:** The environment's transition dynamics and reward model are available. Dynamic programming is a typical example.
- **Model-free methods:** The transition dynamics and reward model are not given. The agent interacts with the environment, collects data, and uses that experience to improve its policy and obtain more reward.

## Q-Tables

A **Q-table** stores the state-action values $Q(s,a)$. A value is not merely the reward for a single step: it estimates the expected future return after taking action $a$ in state $s$ and following a policy thereafter. With a one-step temporal-difference method, we update the table after each transition rather than waiting for the episode to end.

## Model-Free Prediction

### Monte Carlo Policy Evaluation

**Monte Carlo (MC)** methods interact with the environment to sample complete episodes. They estimate the state values of a given policy $\pi$ by averaging the observed returns.

The state-value function

$$
V_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]
$$

is the expected return obtained by starting in state $s$ and following policy $\pi$.

We can therefore execute $\pi$ repeatedly, record the actual return following each visit to $s$, and average those returns. For an episode that terminates at time $T$, the return is

$$
G_t=r_{t+1}+\gamma r_{t+2}+\cdots+\gamma^{T-t-1}r_T.
$$

At the end of each episode, compute the returns for the visited states. For each state $s$, maintain a visit count $N(s)$ and an accumulated return $S(s)$:

- $N(s)\leftarrow N(s)+1$;
- $S(s)\leftarrow S(s)+G_t$;
- $V(s)=S(s)/N(s)$.

When the policy is fixed and the relevant states are sampled sufficiently often, the estimates approach the true values $V_\pi(s)$ as the number of samples increases.

Using the incremental mean formula $\mu_n=\mu_{n-1}+\frac{1}{n}(x_n-\mu_{n-1})$, we can update the value directly without storing the accumulated return:

$$
V(s)\leftarrow V(s)+\frac{1}{N(s)}[G_t-V(s)].
$$

We can also replace the step size $1/N(s)$ with a learning rate $\alpha$:

$$
V(s)\leftarrow V(s)+\alpha[G_t-V(s)].
$$

Here, $G_t-V(s)$ is the error between the observed return and the current estimate. Each update moves $V(s)$ a little closer to the observed return. The step size $1/N(s)$ produces the arithmetic mean; a constant $\alpha$ gives more weight to recent samples.

### Temporal-Difference Learning

One-step temporal-difference learning, **TD(0)**, is the simplest TD method. It bootstraps after every step. The return satisfies $G_t=r_{t+1}+\gamma G_{t+1}$. Although the actual future return $G_{t+1}$ is not yet known, we can replace it with the current estimate $V(s_{t+1})$. This lets us update the value after each transition:

$$
V(s_t)\leftarrow V(s_t)+\alpha[r_{t+1}+\gamma V(s_{t+1})-V(s_t)].
$$

The **TD target** is $r_{t+1}+\gamma V(s_{t+1})$: it combines the immediate reward actually received with the estimated value of the successor state. The **TD error** is the difference between this target and the current estimate:

$$
\delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t).
$$

The update can therefore be written more compactly as $V(s_t)\leftarrow V(s_t)+\alpha\delta_t$. It moves the current estimate toward the TD target. If the next state is terminal, its value is taken to be zero.

We can extend the one-step method to **$n$-step temporal-difference learning**. First observe $n$ steps, then use the estimated value of the later state to account for the remaining return. When those $n$ steps end before termination, the target is

$$
G_t^{(n)}=r_{t+1}+\gamma r_{t+2}+\cdots+\gamma^{n-1}r_{t+n}+\gamma^nV(s_{t+n}).
$$

The update becomes

$$
V(s_t)\leftarrow V(s_t)+\alpha[G_t^{(n)}-V(s_t)].
$$

Larger values of $n$ use more observed rewards. In an episodic task, once the return extends all the way to termination, there is no remaining value to bootstrap from, and the update becomes a Monte Carlo update.

### Comparing Monte Carlo and Temporal-Difference Methods

- TD updates online after each step, whereas MC waits for the episode to finish before updating.
- TD can learn from incomplete trajectories, whereas the MC method described here needs complete episodes.
- TD can learn in continuing tasks without terminal states, whereas episodic MC requires episodes that terminate.
- TD uses the Markov property and summarizes the remaining return with a successor state's estimated value. MC instead uses the actual return from the complete sampled trajectory.

## Model-Free Control

When the environment model is unknown, neither the reward function $R(s,a)$ nor the transition probabilities $P(s'\mid s,a)$ are available for directly computing action values. We can instead use Monte Carlo methods to estimate the action-value function $Q_\pi(s,a)$ from complete episodes collected through interaction with the environment.

**Generalized policy iteration** alternates between two processes. First, fix the current policy $\pi$ and estimate $Q_\pi$ through Monte Carlo sampling. Then improve the policy greedily with respect to the current action values:

$$
\pi'(s)\in\arg\max_a Q_\pi(s,a).
$$

Repeat this process:

$$
\pi\;\longrightarrow\;Q_\pi\;\longrightarrow\;\pi'\;\longrightarrow\;Q_{\pi'}\;\longrightarrow\;\cdots
$$

With suitable exploration and evaluation conditions, the policy and value estimates can approach an optimal policy $\pi^*$ and the optimal action-value function $Q^*$.

To estimate action values using MC, we must sample the relevant state-action pairs $(s,a)$ sufficiently often. One approach is **exploring starts**: begin each episode from a randomly selected state-action pair $(s_0,a_0)$, with every relevant pair having a nonzero probability of being selected. This supplies exploration even when subsequent actions follow a deterministic greedy policy.

<figure>
  <img src="/assets/img/posts/tabular-methods/monte-carlo-exploring-starts.png" alt="Original Chinese pseudocode for Monte Carlo control with exploring starts: initialize a policy and action values, sample episodes from random state-action pairs, average returns, and improve the policy greedily." style="display: block; width: min(100%, 760px); height: auto; margin: 1.5rem auto 0.5rem;" />
  <figcaption>Monte Carlo control with exploring starts (original Chinese pseudocode).</figcaption>
</figure>

In the first-visit version, update a state-action pair only if it **has not appeared earlier in the episode**. This is the intended first-visit condition in the pseudocode above.

To balance exploration and exploitation without requiring exploring starts, we can use an **$\varepsilon$-greedy policy**. With probability $1-\varepsilon$, choose an action with the largest Q-value; with probability $\varepsilon$, choose uniformly at random from all available actions. Explore more early in training, then gradually reduce $\varepsilon$ to rely more on existing experience.

Monte Carlo control repeatedly samples a complete episode using the $\varepsilon$-greedy policy, updates the Q-table from the observed returns, and improves the policy using the updated table. For the same fixed $\varepsilon$, if $Q_\pi$ is evaluated accurately, replacing an $\varepsilon$-greedy policy with a policy $\pi'$ that is $\varepsilon$-greedy with respect to $Q_\pi$ guarantees

$$
V_{\pi'}(s)\ge V_\pi(s).
$$

<figure>
  <img src="/assets/img/posts/tabular-methods/monte-carlo-epsilon-greedy.png" alt="Original Chinese pseudocode for epsilon-greedy Monte Carlo control: initialize action values and counts, sample episodes, update action values with incremental means, reduce epsilon, and update the epsilon-greedy policy." style="display: block; width: min(100%, 760px); height: auto; margin: 1.5rem auto 0.5rem;" />
  <figcaption>Monte Carlo control with epsilon-greedy exploration (original Chinese pseudocode).</figcaption>
</figure>
