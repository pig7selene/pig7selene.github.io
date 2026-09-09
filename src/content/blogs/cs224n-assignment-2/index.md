---
title: "CS224N | Assignment 2"
publishDate: 2026-08-25
category: learning
tags: [NLP, CS224N]
language: en
description: "personal solution to CS224N assignment 2"
heroImage:
  src: /assets/img/posts/cs224n-assignment-2/assignment-2-card-cover.png
  alt: "Moonlit cityscape with a silver-haired figure"
  color: "#5273a3"
---

You can download the [LaTeX](/output/pdf/cs224n-assignment-2.tex) and [PDF](/output/pdf/cs224n-assignment-2.pdf).

## Problem 1

### (a)

Let y \in \mathbb{R}^{\lvert V\rvert} be the one-hot true distribution, let \hat{y} \in \mathbb{R}^{\lvert V\rvert} be the predicted distribution, and let

$$
\hat{y}\_w = P(O=w\mid C=c) = \frac{\exp(u\_w^\top v\_c)} {\sum\_{j\in V}\exp(u\_j^\top v\_c)}.
$$

The matrix U contains the outside-word vectors as its columns:

$$
U = \begin{bmatrix}u\_1 & u\_2 & \cdots & u\_{\lvert V\rvert}\end{bmatrix}.
$$

Since y is one-hot, y\_o=1 and y\_w=0 for every w\neq o. Therefore,

$$
-\sum\_{w\in V}y\_w\log \hat{y}\_w = -\log \hat{y}\_o.
$$

### (b)

**(i)**

First, rewrite the naive-softmax loss as

$$
J\_{\text{naive-softmax}}(v\_c,o,U) = -\log \hat{y}\_o = -u\_o^\top v\_c+ \log\left(\sum\_{w\in V}\exp(u\_w^\top v\_c)\right).
$$

Taking the derivative with respect to v\_c,

$$
\frac{\partial J}{\partial v\_c} = -u\_o+ \frac{\sum\_{w\in V}\exp(u\_w^\top v\_c)u\_w} {\sum\_{j\in V}\exp(u\_j^\top v\_c)} = -u\_o + \sum\_{w\in V}\hat{y}\_w u\_w.
$$

Because y is one-hot,

$$
u\_o = Uy,\quad \sum\_{w\in V}\hat{y}\_w u\_w = U\hat{y}.
$$

Hence the vectorized result is

$$
\frac{\partial J}{\partial v\_c}=U(\hat{y}-y)
$$

**(ii)**

The gradient is zero exactly when

$$
U(\hat{y}-y)=0
$$

Equivalently, \hat{y}-y\in\operatorname{Null}(U). In particular,\hat{y}=y is sufficient for the gradient to be zero.

**(iii)**

Using a learning rate \eta>0, gradient descent gives

$$
v\_c \leftarrow v\_c-\eta\left(U\hat{y}-Uy\right) =v\_c+\eta u\_o-\eta\sum\_{w\in V}\hat{y}\_w u\_w.
$$

The +\eta u\_o term moves the center-word vector v\_c toward the true outside-word vector u\_o, increasing their similarity. The -\eta\sum\_w\hat{y}\_w u\_w term moves v\_c away from the probability-weighted average of the predicted outside-word vectors, with a larger correction for words to which the model currently assigns more probability. Together, these changes increase the score of the true outside word relative to the scores of the other words.

### (c)

Suppose u\_x=\alpha u\_y for x\neq y. If \alpha>0, then

$$
\frac{u\_x}{\lVert u\_x\rVert\_2} =\frac{\alpha u\_y}{\lVert \alpha u\_y\rVert\_2} =\frac{u\_y}{\lVert u\_y\rVert\_2}.
$$

Thus, L2 normalization makes two vectors that point in the same direction identical, even if their original magnitudes differ. It removes useful information when vector magnitude encodes information relevant to the phrase classification—for example, when x and y have similar semantic directions but different strengths or polarities in the classifier. It does not remove useful information when only vector direction matters, or when same-direction vectors with different magnitudes should have the same effect on the final classification.

### (d)

Since U is formed by placing the outside-word vectors in columns, its
gradient has the same column structure:

$$
\frac{\partial J(v\_c,o,U)}{\partial U}= \begin{bmatrix} \dfrac{\partial J(v\_c,o,U)}{\partial u\_1} & \dfrac{\partial J(v\_c,o,U)}{\partial u\_2} & \cdots & \dfrac{\partial J(v\_c,o,U)}{\partial u\_{\lvert V\rvert}} \end{bmatrix}
$$

### (e)

The loss is

$$
J=-u\_o^\top v\_c+\log\left(\sum\_{j\in V}\exp(u\_j^\top v\_c)\right).
$$

For the true outside word w=o,

$$
\frac{\partial J}{\partial u\_o} =-v\_c+\frac{\exp(u\_o^\top v\_c)}{\sum\_{j\in V}\exp(u\_j^\top v\_c)}v\_c =(\hat{y}\_o-1)v\_c.
$$

For every w\neq o,

$$
\frac{\partial J}{\partial u\_w} =\frac{\exp(u\_w^\top v\_c)}{\sum\_{j\in V}\exp(u\_j^\top v\_c)}v\_c =\hat{y}\_w v\_c.
$$

Therefore,

$$
\frac{\partial J}{\partial u\_w} =(\hat{y}\_w-y\_w)v\_c= \begin{cases} (\hat{y}\_o-1)v\_c, & w=o,\\\ \hat{y}\_w v\_c, & w\neq o. \end{cases}
$$

### (f)

For f(x)=\max(\alpha x,x), where 0<\alpha<1,

$$
f'(x)= \begin{cases} 1, & x>0,\\\ \alpha, & x<0. \end{cases}
$$

The derivative at x=0 is not required.

### (g)

Starting from

$$
\sigma(x)=\frac{1}{1+e^{-x}},
$$

we obtain

$$
\sigma'(x) =\frac{e^{-x}}{(1+e^{-x})^2} =\frac{1}{1+e^{-x}}\left(1-\frac{1}{1+e^{-x}}\right).
$$

Hence,

$$
\sigma'(x)=\sigma(x)\bigl(1-\sigma(x)\bigr)
$$

## Problem 2

### (a)

**(i)**

Momentum averages gradients over multiple minibatches, so random variations in individual minibatch gradients tend to cancel out. Directions that are consistently useful accumulate in m, making parameter updates smoother and less noisy. This can reduce oscillation and allow the model to converge more stably and quickly.

**(ii)**

Parameters whose gradients have historically been small receive relatively larger updates, since their corresponding values in v are smaller. Parameters with consistently large gradients receive smaller updates. This adaptive scaling prevents large-gradient parameters from changing too aggressively while still allowing small-gradient parameters to make meaningful progress.

### (b)

**(i)**

For each hidden unit $i$,

$$
(h_{\text{drop}})_i=\gamma d_i h_i.
$$

Since $\mathbb{E}[d_i]=1-p_{\text{drop}}$,

$$
\mathbb{E}[(h_{\text{drop}})_i]=
\gamma h_i(1-p_{\text{drop}}).
$$

Requiring $\mathbb{E}[(h_{\text{drop}})_i]=h_i$ gives

$$
\gamma=\frac{1}{1-p_{\text{drop}}}.
$$

**(ii)**

Dropout is applied during training as a regularization method, since randomly dropping hidden units prevents the model from relying too heavily on specific neurons and reduces overfitting. During evaluation, dropout is disabled so that the full network is used and predictions remain stable and deterministic.
