---
title: "矩阵树定理的两种证明"
publishDate: 2026-06-25
category: learning
tags: [graph-theory]
language: zh
description: "Two proofs of the Matrix-Tree Theorem—by induction and Cauchy-Binet—showing how determinants count graph spanning trees."
heroImage:
  src: /assets/img/posts/matrix-tree/matrix-tree-card-cover-v2.png
  alt: "文章封面"
  color: "#3287bd"
---

<aside class="download-callout" role="note">
  <p>You can download the <a href="/output/pdf/two-proofs-of-the-matrix-tree-theorem.tex">LaTeX</a> and <a href="/output/pdf/two-proofs-of-the-matrix-tree-theorem.pdf">PDF</a>.</p>
</aside>

本文介绍矩阵树定理，并分别使用归纳法和 Cauchy–Binet 公式给出两种证明，说明如何通过行列式计算图的生成树数量。

用 $A[i]$ 表示从矩阵 $A$ 中删除第 $i$ 行与第 $i$ 列得到的矩阵。

## Kirchhoff 矩阵树定理

**定理（Kirchhoff 矩阵树定理）。**

图 $G$ 的生成树数量为

$$
\tau(G)=\det(L_G[i]),
$$

其中，$i$ 可以是任意顶点。

这里，$L_G$ 是图 $G$ 的拉普拉斯矩阵。

## Fact 1

设 $A \in \mathbb{R}^{n \times n}$，$E_{ii}$ 表示仅在 $(i,i)$ 位置为 $1$、其余位置均为 $0$ 的矩阵，则

$$
\det(A + E_{ii})
=
\det(A) + \det(A[i]).
$$

考虑行列式的排列展开定义：

$$
\det(A=(a_{ij}))
=
\sum_{\pi}
\operatorname{sgn}(\pi)
a_{1\pi(1)}
a_{2\pi(2)}
\cdots
a_{n\pi(n)}.
$$

这个结论不难看出。将 $(i,i)$ 位置的元素替换为 $a_{ii}+1$ 后，得到原行列式加上一个额外项。

这个额外项恰好对应集合 $[n]\setminus\{i\}$ 上所有排列的求和，也就是删除第 $i$ 行与第 $i$ 列后得到的行列式，因此它就是 $\det(A[i])$。

## First Proof

下面开始证明 Kirchhoff 矩阵树定理。

第一种证明同时对图 $G$ 的顶点数和边数进行归纳。

如果 $G$ 是一个只有两个顶点、没有边的图，则

$$
L_G=
\begin{bmatrix}
0 & 0\\
0 & 0
\end{bmatrix}.
$$

因此，$L_G[i]=[0]$，并且

$$
\det(L_G[i])=0,
$$

与定理一致。

证明中使用以下记号：

- $\tau(G)$ 表示图 $G$ 的生成树数量；
- $G-e$ 表示从图中删除边 $e$；
- $G/e$ 表示收缩边 $e$，即将它的两个端点合并为一个顶点。

如果 $i$ 是孤立点，图 $G$ 不存在生成树，$L_G$ 的第 $i$ 行与第 $i$ 列都为零。因此

$$
\det(L_G[i])=\det(L_{G-i})=0.
$$

否则，可以假设存在一条与顶点 $i$ 相邻的边 $e=(i,j)$。对于任意生成树 $T$，要么 $e \in T$，要么 $e \notin T$，所以

$$
\tau(G)=\tau(G/e)+\tau(G-e).
$$

第一项将顶点数减少 $1$，第二项将边数减少 $1$，由此建立归纳框架。

先建立 $L_G$ 与 $L_{G-e}$ 的关系。注意到

$$
L_G[i]=L_{G-e}[i]+E_{jj}.
$$

也就是说，删除边 $e$ 后，得到的矩阵与 $L_G[i]$ 的差异只来自点 $j$ 的度数。利用结论 1，得到

$$
\begin{aligned}
\det(L_G[i])
&= \det(L_{G-e}[i]+E_{jj}) \\
&= \det(L_{G-e}[i])+\det(L_{G-e}[i,j]) \\
&= \det(L_{G-e}[i])+\det(L_G[i,j]).
\end{aligned}
$$

其中，$L_G[i,j]$ 表示同时删除第 $i$、$j$ 行与列后得到的矩阵。最后一个等号成立，是因为移除 $i,j$ 所在的行和列后，$L_G$ 与 $L_{G-e}$ 已没有差异。

再建立 $L_G$ 与 $L_{G/e}$ 的关系。设将点 $i$ 收缩到点 $j$，则

$$
L_{G/e}[j]=L_G[i,j].
$$

因此，

$$
\begin{aligned}
\det(L_G[i])
&= \det(L_{G-e}[i])+\det(L_{G/e}[j]) \\
&= \tau(G-e)+\tau(G/e) \\
&= \tau(G).
\end{aligned}
$$

第二个等号来自归纳假设，第一种证明完成。

## Fact 2: Cauchy–Binet 公式

设

$$
A \in \mathbb{R}^{n \times m},
\qquad
B \in \mathbb{R}^{m \times n},
$$

其中 $m \ge n$。定义：

- $A_S$：从 $A$ 中取出由 $S\subseteq[m]$ 指定的列组成的子矩阵；
- $B_S$：从 $B$ 中取出由 $S\subseteq[m]$ 指定的行组成的子矩阵。

用 $\binom{[m]}{n}$ 表示 $[m]$ 中所有大小为 $n$ 的子集组成的集合，则

$$
\det(AB)
=
\sum_{S \in \binom{[m]}{n}}
\det(A_S)\det(B_S).
$$

## Second Proof

由于

$$
L_G = \sum_{(i,j) \in E}(e_i - e_j)(e_i-e_j)^T,
$$

可以写成

$$
L_G=BB^T,
$$

其中，$B \in \mathbb{R}^{n \times m}$ 为图 $G$ 的一个关联矩阵。每条边 $(i,j)$ 对应一列向量

$$
e_i-e_j.
$$

由于 $L_G=BB^T$，也再次说明 $L_G$ 是半正定矩阵。

用 $B[i]$ 表示从 $B$ 中删除第 $i$ 行得到的矩阵，则

$$
L_G[i]=B[i]B[i]^T.
$$

进一步用 $B_S[i]$ 表示从 $B[i]$ 中只保留由 $S\subseteq E$ 指定的列得到的矩阵。

我们还需要下面的引理。

**引理 2。**

对于满足 $\lvert S\rvert = n-1$ 的 $S \subseteq E$，

$$
\left|\det(B_S[i])\right|
=
\begin{cases}
1, & \text{if } S \text{ is a spanning tree},\\
0, & \text{otherwise}.
\end{cases}
$$

有了这个引理，矩阵树定理的第二种证明非常简短：

$$
\begin{aligned}
\det(L_G[i])
&= \det(B[i]B[i]^T) \\
&= \sum_{S\in\binom{E}{n-1}}
   \det(B_S[i])^2 \\
&= \tau(G).
\end{aligned}
$$

其中：

- 第二个等号来自 Cauchy–Binet 公式；
- 第三个等号来自引理 2。

## Proof of Lemma 2

不失一般性，给 $B_S[i]$ 中的边任意指定方向。也就是说，可以将边 $(i,j)$ 对应的列从 $e_i-e_j$ 改为 $e_j-e_i$。

这样只会改变行列式的符号，不会改变其绝对值。

若 $S\subseteq E$、$\lvert S\rvert=n-1$，且 $S$ 不是生成树，则 $S$ 必然包含一个环。可以将环上各边的方向沿环统一。

将这个环对应的各列相加，得到零向量。因此，$B_S[i]$ 的列线性相关，从而

$$
\det(B_S[i])=0.
$$

现在设 $S$ 是一棵生成树。下面通过归纳证明

$$
\left|\det(B_S[i])\right|=1
$$

其中归纳变量为 $n$。

### 基础情况

若 $n=2$，则

$$
B_S=
\begin{bmatrix}
1\\
-1
\end{bmatrix}.
$$

删除一行后，得到

$$
B_S[i]=[\pm 1],
$$

因此，

$$
\left|\det(B_S[i])\right|=1.
$$

### 归纳步骤

假设引理对有 $n-1$ 个顶点的树成立。

取树中一个叶子节点 $j\ne i$，并令 $(k,j)$ 为与 $j$ 相邻的唯一一条边。这样的叶子节点一定存在，因为每棵至少有两个顶点的树都至少有两个叶子。

对行与列进行置换，使得：

- 边 $(k,j)$ 成为最后一列；
- 点 $j$ 成为最后一行。

这些置换可能改变行列式的符号，但不影响其绝对值。

由于 $j$ 是叶子，最后一行只有一个非零元素。沿最后一行展开行列式，得到

$$
\left|\det(B_S[i])\right|
=
\left|
\det(B_{S-\{(k,j)\}}[i,j])
\right|.
$$

由归纳假设，

$$
\left|
\det(B_{S-\{(k,j)\}}[i,j])
\right|
=1.
$$

因此，

$$
\left|\det(B_S[i])\right|=1.
$$

引理 2 得证，Kirchhoff 矩阵树定理的第二种证明也随之完成。
