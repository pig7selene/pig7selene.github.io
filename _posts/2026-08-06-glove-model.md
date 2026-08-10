---
title: "GloVe Model"
date: 2026-08-06 16:35:00 +0800
categories: [Machine Learning]
tags: [NLP]
math: true
description: "CS224n lec2,the GloVe model, co-occurrence matrices, least-squares objectives, and the evaluation of word vectors."
---

> from Selene, Chinese version: [Chinese version](https://notes.sjtu.edu.cn/s/SW1CcdBqW)

# GloVe Model

## 1.1 Comparison with Previous Models

Previously, we studied the skip-gram model, which mainly learns word embeddings by making predictions within local context windows. It can capture complex linguistic patterns beyond word similarity, but it does not make use of global co-occurrence statistics.

The GloVe model, on the other hand, uses global statistical information and predicts the probability that word $j$ appears in the context of word $i$ through a least-squares objective.

## 1.2 Co-occurrence Matrix

Let $X$ be the word-word co-occurrence matrix, where $X_{ij}$ denotes the number of times word $j$ appears in the context of word $i$

Let

$$
X_i = \sum_k X_{ik}
$$

denote the total number of times any word $k$ appears in the context of word $i$

Finally, let

$$
P_{ij}=P(w_j \mid w_i)=\frac{X_{ij}}{X_i}
$$

denote the probability that word $j$ appears in the context of word $i$.

## 1.3 Least-Squares Objective Function

In the skip-gram model, we use softmax to compute the probability that word $j$ appears in the context of word $i$:

$$
Q_{ij} = \frac{exp(u_{j}^{T}x_i)}{\sum_{w=1}^{W}exp(u_w^T v_i)}
$$

Training is performed in an online and stochastic manner, using the following global cross-entropy loss function:

$$
J = -\sum_{j \in corpus}\sum_{j \in context(i)} logQ_{ij}
$$

A significant disadvantage of cross-entropy is that it requires the distribution $Q$ to be properly normalized, which requires an expensive summation over the entire vocabulary. For this reason, we instead use a least-squares objective and discard the normalization factors in $P$ and $Q$:

$$
\hat{J}=\sum_{i=1}^{W}\sum_{j=1}^{W}X_i\left(\hat{p}_{ij}-\hat{Q}_{ij}\right)^2
$$

where

$$
\hat{P}_{ij}=X_{ij},\qquad
\hat{Q}_{ij}=\exp\left(u_j^{T}v_i\right)
$$

are both unnormalized distributions. Since $X_{ij}$ is often very large, optimization becomes difficult. An effective modification is to minimize the squared error between the logarithms of $\hat{P}$ and $\hat{Q}$:

$$
\hat{J}=\sum_{i=1}^{W}\sum_{j=1}^{W}X_i\left(\log\hat{P}_{ij}-\log\hat{Q}_{ij}\right)^2=\sum_{i=1}^{W}\sum_{j=1}^{W}X_i\left(u_j^{T}v_i-\log X_{ij}\right)^2
$$

# Evaluation of Word Vectors

## 2.1 Intrinsic Evaluation

Intrinsic evaluation of word vectors evaluates a set of word vectors produced by an embedding method on a specific intermediate subtask. Such subtasks are usually simple and fast to compute, so they can help us understand the system that generates the word vectors. Intrinsic evaluation should usually return a numerical value that indicates how well these word vectors perform on the evaluation subtask.

## 2.2 Extrinsic Evaluation

Extrinsic evaluation of word vectors evaluates a set of word vectors produced by an embedding method on the real task at hand. Such tasks are usually more complex and slower to compute. In general, optimizing only for a poorly performing extrinsic evaluation system cannot determine exactly which specific subsystem has gone wrong. This also illustrates the necessity of intrinsic evaluation.
