---
title: "CS224N | Assignment 2"
date: 2026-08-25 23:25:00 +0800
categories: [Machine Learning]
tags: [NLP,CS224N]
math: true
description: "personal solution to CS224N assignment 2"
---

## Problem 1

Let $y \in \mathbb{R}^{|V|}$ be the one-hot true distribution, let$\hat{y} \in \mathbb{R}^{|V|}$ be the predicted distribution, and let

$$
\hat{y}_w
= P(O=w\mid C=c)
= \frac{\exp(u_w^\top v_c)}
{\sum_{j\in V}\exp(u_j^\top v_c)}.
$$

The matrix $U$ contains the outside-word vectors as its columns:

$$
U = \begin{bmatrix}u_1 & u_2 & \cdots & u_{|V|}\end{bmatrix}.
$$

### (a)

Since $y$ is one-hot, $y_o=1$ and $y_w=0$ for every $w\neq o$. Therefore,

$$
-\sum_{w\in V}y_w\log \hat{y}_w
= -\log \hat{y}_o.
$$

### (b)

#### (i) Derivative with respect to $v_c$

First, rewrite the naive-softmax loss as

$$
\begin{aligned}
J_{\text{naive-softmax}}(v_c,o,U)&= -\log \hat{y}_o \\&= -u_o^\top v_c+ \log\left(\sum_{w\in V}\exp(u_w^\top v_c)\right).
\end{aligned}
$$

Taking the derivative with respect to $v_c$,

$$
\begin{aligned}
\frac{\partial J}{\partial v_c}&= -u_o+ \frac{\sum_{w\in V}\exp(u_w^\top v_c)u_w}
          {\sum_{j\in V}\exp(u_j^\top v_c)} \\&= -u_o + \sum_{w\in V}\hat{y}_w u_w.
\end{aligned}
$$

Because $y$ is one-hot,

$$
u_o = Uy,\quad
\sum_{w\in V}\hat{y}_w u_w = U\hat{y}.
$$

Hence the vectorized result is

$$
\frac{\partial J}{\partial v_c}=U(\hat{y}-y)
$$

#### (ii) When the gradient is zero

The gradient is zero exactly when

$$
U(\hat{y}-y)=0
$$

Equivalently, $\hat{y}-y\in\operatorname{Null}(U)$. In particular,$\hat{y}=y$ is sufficient for the gradient to be zero.

#### (iii) Interpretation of the two terms

Using a learning rate $\eta>0$, gradient descent gives

$$
\begin{aligned}
v_c
&\leftarrow v_c-\eta\left(U\hat{y}-Uy\right) \\
&=v_c+\eta u_o-\eta\sum_{w\in V}\hat{y}_w u_w.
\end{aligned}
$$

The $+\eta u_o$ term moves the center-word vector $v_c$ toward the trueoutside-word vector $u_o$, increasing their similarity. The$-\eta\sum_w\hat{y}_w u_w$ term moves $v_c$ away from the probability-weighted average of the predicted outside-word vectors, with a larger correction for words to which the model currently assigns more probability. Together, these changes increase the score of the true outside word relative to the scores of the other words.

### (c)

Suppose $u_x=\alpha u_y$ for $x\neq y$.If $\alpha>0$, then

$$
\frac{u_x}{\lVert u_x\rVert_2}
=\frac{\alpha u_y}{\lVert \alpha u_y\rVert_2}
=\frac{u_y}{\lVert u_y\rVert_2}.
$$

Thus, L2 normalization makes two vectors that point in the same direction identical, even if their original magnitudes differ. It removes useful information when vector magnitude encodes information relevant to the phrase classification—for example, when $x$ and $y$ have similar semantic directions but different strengths or polarities in the classifier. It does not remove useful information when only vector direction matters, or when same-direction vectors with different magnitudes should have the same effect on the final classification.

### (d)

Since $U$ is formed by placing the outside-word vectors in columns, its
gradient has the same column structure:

$$
\frac{\partial J(v_c,o,U)}{\partial U}=
\begin{bmatrix}
\dfrac{\partial J(v_c,o,U)}{\partial u_1} &
\dfrac{\partial J(v_c,o,U)}{\partial u_2} &
\cdots &
\dfrac{\partial J(v_c,o,U)}{\partial u_{|V|}}
\end{bmatrix}
$$

### (e)

The loss is

$$
J=-u_o^\top v_c+\log\left(\sum_{j\in V}\exp(u_j^\top v_c)\right).
$$

For the true outside word $w=o$,

$$
\begin{aligned}
\frac{\partial J}{\partial u_o}
&=-v_c+\frac{\exp(u_o^\top v_c)}{\sum_{j\in V}\exp(u_j^\top v_c)}v_c \\
&=(\hat{y}_o-1)v_c.
\end{aligned}
$$

For every $w\neq o$,

$$
\begin{aligned}
\frac{\partial J}{\partial u_w}
&=\frac{\exp(u_w^\top v_c)}{\sum_{j\in V}\exp(u_j^\top v_c)}v_c \\
&=\hat{y}_w v_c.
\end{aligned}
$$

Therefore,

$$
\frac{\partial J}{\partial u_w}
=(\hat{y}_w-y_w)v_c=
\begin{cases}
(\hat{y}_o-1)v_c, & w=o,\\
\hat{y}_w v_c, & w\neq o.
\end{cases}
$$

### (f)

For $f(x)=\max(\alpha x,x)$, where $0<\alpha<1$,

$$
f'(x)=
\begin{cases}
1, & x>0,\\
\alpha, & x<0.
\end{cases}
$$

The derivative at $x=0$ is not required.

### (g)

Starting from

$$
\sigma(x)=\frac{1}{1+e^{-x}},
$$

we obtain

$$
\begin{aligned}
\sigma'(x)
&=\frac{e^{-x}}{(1+e^{-x})^2} \\
&=\frac{1}{1+e^{-x}}\left(1-\frac{1}{1+e^{-x}}\right).
\end{aligned}
$$

Hence,

$$
\sigma'(x)=\sigma(x)\bigl(1-\sigma(x)\bigr)
$$


## Problem 2

### (a)

#### (i)

Momentum averages gradients over multiple minibatches, so random variations in individual minibatch gradients tend to cancel out. Directions that are consistently useful accumulate in m, making parameter updates smoother and less noisy. This can reduce oscillation and allow the model to converge more stably and quickly.

#### (ii)

Parameters whose gradients have historically been small receive relatively larger updates, since their corresponding values in v are smaller. Parameters with consistently large gradients receive smaller updates. This adaptive scaling prevents large-gradient parameters from changing too aggressively while still allowing small-gradient parameters to make meaningful progress.
