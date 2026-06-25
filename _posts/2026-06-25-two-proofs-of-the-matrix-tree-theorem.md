---
title: "Two Proofs of the Matrix-Tree Theorem"
date: 2026-06-25 15:30:00 +0800
categories: [Mathematics, Graph Theory]
tags: [graph-theory]
math: true
description: "This chapter introduces the Matrix-Tree Theorem and presents two proofs, using induction and the Cauchy-Binet formula, to show how determinants can be used to count spanning trees in a graph."
---

# Matrix-Tree Theorem

We denote by $A[i]$ the matrix obtained from $A$ by deleting the $i$-th row and the $i$-th column.

**Kirchhoff’s Matrix-Tree Theorem**

The number of spanning trees of a graph $G$ is given by $\det(L_G[i])$, where $i$ can be any vertex.

**Fact 1**

Let $A \in \mathbb{R}^{n \times n}$,
and let $E_{ii}$ be the matrix whose $(i,i)$ entry is $1$ and all other entries are $0$. Then

$$
\det(A + E_{ii})
=
\det(A) + \det(A[i])
$$

Consider the permutation expansion definition of the determinant:

$$
\det(A=(a_{ij}))
=
\sum_{\pi}
\operatorname{sgn}(\pi)
a_{1\pi(1)}
a_{2\pi(2)}
\dots
a_{n\pi(n)}
$$

This fact is not difficult to see. We replace the $(i,i)$ entry by $a_{ii}+1$, so we obtain the original determinant plus one extra term.

This extra term corresponds exactly to the sum over all permutations on the set $[n]\backslash\{i\}$, namely the determinant obtained by avoiding the $i$-th row and $i$-th column. Hence this term is precisely $\det(A[i])$.

**Proof**

Now we begin the proof of **Kirchhoff’s Matrix-Tree Theorem**.

Our first proof proceeds by induction on both the number of vertices and the number of edges of the graph $G$.

If $G$ is an empty graph with two vertices, then

$$
L_G=
\begin{bmatrix}
0 & 0\\
0 & 0
\end{bmatrix}
$$

Hence $L_G[i]=[0]$, and $\det(L_G[i])=0$, which agrees with the theorem.

Throughout the proof:

- $\tau(G)$ denotes the number of spanning trees of $G$;
- $G-e$ denotes the graph obtained by deleting the edge $e$;
- $G/e$ denotes the graph obtained by contracting the edge $e$ (that is, merging its two endpoints into a single vertex).

If $i$ is an isolated vertex, then $G$ has no spanning trees, and the $i$-th row and $i$-th column of $L_G$ are entirely zero. Therefore

$$
\det(L_G[i])=\det(L_{G-i})=0
$$

Otherwise, we may assume there exists an edge $e=(i,j)$ adjacent to $i$. For any spanning tree $T$, either $e \in T$ or $e \notin T$, so

$$
\tau(G)=\tau(G/e)+\tau(G-e)
$$

The first term reduces the number of vertices by $1$, while the second reduces the number of edges by $1$, so this gives us an induction framework.

First, we establish the relation between $L_G$ and $L_{G-e}$. Observe that

$$
L_G[i]=L_{G-e}[i]+E_{jj}.
$$

That is, after deleting the edge $e$, the only difference between the resulting Laplacian matrix and $L_G[i]$ lies in the degree of vertex $j$. By Fact 1, we obtain

$$
\begin{aligned}
\det(L_G[i])
&= \det(L_{G-e}[i]+E_{jj}) \\
&= \det(L_{G-e}[i])+\det(L_{G-e}[i,j]) \\
&= \det(L_{G-e}[i])+\det(L_G[i,j]).
\end{aligned}
$$

Here $L_G[i,j]$ denotes the matrix obtained by deleting both the $i$-th and $j$-th rows and columns. The last equality holds because once the $i,j$ rows and columns are removed, there is no remaining difference between $L_G$ and $L_{G-e}$.

Next, we establish the relation between $L_G$ and $L_{G/e}$. Suppose we contract vertex $i$ into vertex $j$ (so $L_{G/e}$ no longer contains the row or column corresponding to $i$). Then

$$
L_{G/e}[j]=L_G[i,j]
$$

Hence

$$
\begin{aligned}
\det(L_G[i])
    &= \det(L_{G-e}[i])+\det(L_{G/e}[j]) \\
    &= \tau(G-e)+\tau(G/e) \\
    &= \tau(G).
\end{aligned}
$$

The second equality follows from the induction hypothesis, completing the proof.


**Fact 2 (Cauchy-Binet Formula)**

Let

$$
A \in \mathbb{R}^{n \times m},
\quad
B \in \mathbb{R}^{m \times n}
$$

where $m \ge n$. Define:

- $A_S$: the submatrix of $A$ consisting of the columns indexed by $S\subseteq[m]$;
- $B_S$: the submatrix of $B$ consisting of the rows indexed by $S\subseteq[m]$.

Let $\binom{[m]}{n}$ denote the collection of all subsets of $[m]$ of size $n$. Then

$$
\det(AB)
=
\sum_{S \in \binom{[m]}{n}}
\det(A_S)\det(B_S)
$$

Since

$$
L_G = \sum_{(i,j) \in E}(e_i - e_j)(e_i-e_j)^T,
$$

we may write $L_G=BB^T$, where $B \in \mathbb{R}^{m \times n}$. For each edge $(i,j)$, there is a corresponding column vector

$$
e_i-e_j
$$

Since $L_G=BB^T$, this once again proves that $L_G$ is positive semidefinite.

If $B[i]$ denotes the matrix obtained by deleting the $i$-th row, then

$$
L_G[i]=B[i]B[i]^T
$$

We further let $B_S[i]$ denote the matrix obtained from $B[i]$ by keeping only the columns indexed by $S\subseteq E$.

We also need the following lemma, whose proof will be given later.

**Lemma 2**

For $S \subseteq E$ with $|S|=n-1$,

$$
\left|\det(B_S[i])\right|
=
\begin{cases}
1, & \text{if } S \text{ is a spanning tree},\\
0, & \text{otherwise}.
\end{cases}
$$

With this lemma, the second proof of the Matrix-Tree Theorem becomes extremely short:

$$
\begin{aligned}
\det(L_G[i])
    &= \det(B[i]B[i]^T) \\
    &= \sum_{S\in\binom{E}{n-1}}
       \left|\det(B_S[i])\right|
       \left(\det(B_S[i])\right) \\
    &= \tau(G).
\end{aligned}
$$

Here:

- the second equality follows from the Cauchy-Binet formula;
- the third equality follows from Lemma 2.

**Proof of Lemma 2**

Assume without loss of generality that the edges in $B_S[i]$ are oriented arbitrarily. That is, we may replace the column corresponding to $(i,j)$ from $e_i-e_j$ to $e_j-e_i$.

This only changes the sign of the determinant. If $S\subseteq E$, $|S|=n-1$, and $S$ is not a spanning tree, then it necessarily contains a cycle. We may orient the edges along this cycle.

If we sum the columns corresponding to this cycle, we obtain the zero vector. Hence the columns of $B_S[i]$ are linearly dependent, and therefore

$$
\det(B_S[i])=0
$$

Now assume that $S$ is a spanning tree, and proceed by induction on $n$.

**Base case**

If $n=2$, then

$$
B_S=
\begin{bmatrix}
1\\
-1
\end{bmatrix},
$$

hence $B_S[i]=\pm1$, and therefore

$$
\det(B_S[i])=1
$$

**Inductive case**

Assume the lemma holds for graphs of size $n-1$. Let $j\neq i$ be a leaf vertex in the tree, and let $(k,j)$ be the unique edge adjacent to $j$.

Permute rows and columns so that:

- $(k,j)$ becomes the last column;
- $j$ becomes the last row.

This may change the sign of the determinant, but that is irrelevant. Then

$$
B_S[i]=
\begin{bmatrix}
 &  &  & 0\\
 &  &  & 0\\
 & B_{S-\{(k,j)\}}[i,j] &  & 1\\
 &  &  & 0\\
0 & \cdots & 0 & -1
\end{bmatrix}
$$

Expanding the determinant along the last row gives

$$
\left|\det(B_S[i])\right|
=
\left|
\det(B_{S-\{(k,j)\}}[i,j])
\right|
=
1.
$$

The last equality follows from the induction hypothesis, since $S-\{(k,j)\}$ is a tree on the vertex set obtained after deleting vertex $j$, and we assumed that $i$ is not the leaf vertex.
