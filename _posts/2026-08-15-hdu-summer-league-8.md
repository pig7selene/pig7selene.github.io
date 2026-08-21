---
title: "HDU Summer League (8)"
date: 2026-08-15 22:09:00 +0800
categories: [xcpc]
tags: [xcpc,hdu summer league2026]
math: true
description: "Solutions and notes for HDU Summer League(8)."
---

## HDU Summer League(8)
> [Code Implementation for This Contest](https://github.com/pig7selene/xcpc-code/tree/main/summer08)  
> from Selene, Chinese version: [Chinese version](https://notes.sjtu.edu.cn/s/WTuny4iBr)

### 1001

**Problem**: Given a directed graph with $n$ vertices and $m$ edges, you need to travel from vertex $1$ to vertex $n$. There are $q$ operations. The $i$-th operation deletes all outgoing edges of vertex $p_i$. You need to find the maximum number of operations $k$ such that after performing the first $k$ operations, there is still a path from vertex $1$ to vertex $n$.

**Solution**: First assume that all outgoing edges of every $p_i$ have already been deleted. We then add the edges back in reverse order while maintaining two arrays

* $on[u]$: whether the outgoing edges of vertex $u$ currently exist
* $vis[u]$: whether $u$ can currently reach $n$

When restoring vertex $u=p[i]$ in reverse order, set $on[u] = 1$. If there exists an edge $u \to v$ and $vis[v] = 1$, then $vis[u] = 1$. Then perform a bfs starting from $u$ on the reverse graph.

### 1002

**Problem**: Given two sequences $a$ and $b$ of length $n$, there are $m$ days. At the beginning of each day, $a_i = a_i + b_i$ is performed, and there is one query every day

* `1 l r x`: for all $l \le i \le r$, perform $b_i \leftarrow b_i+x$
* `2 l r`: calculate $\sum_{i=l}^{r} a_i \bmod 2^{64}$

**Solution**: Maintain prefix sums for both arrays. Since the modulus is $2^{64}$, we can directly use `ull`. For the first operation, use a Fenwick tree to maintain two lazy arrays `lazy1` and `lazy2`. One stores all modifications $x$, while the other stores $day \times x$. On day $t$, the answer for an interval $[l,r]$ is

$$
\sum A_i+t\sum B_i+t\sum lazy1_i-\sum lazy2_i
$$

### 1003

**Problem**: Given a permutation, each query gives `x k`. We need to select some positions such that $x$ is not selected and the distance between any two selected positions is at least $k$. Find the number of valid selection schemes.

**Solution**: We can think about it in reverse and calculate the total number of schemes minus the number of schemes in which $x$ is selected.

Let $f(n,k)$ denote the number of schemes for $n$ positions where the distance between any two selected positions is at least $k$. For a query $(x,k)$, the total number of schemes is $f(n,k)$. For schemes that must contain $x$, consider the valid positions on the left and right of $x$. The valid length on the left is $l=\max(0,x-k)$, and the valid length on the right is $r=\max(0,n-x-k+1)$. Therefore, the answer is

$$
f(n,k)-f(l,k)f(r,k)
$$

There are two ways to calculate $f$:

Recurrence:

$$
f(i,k)=f(i-1,k)+f(i-k,k)
$$

For a fixed $k$, it can be precomputed in $O(n)$.

Or enumerate the number $c$ of selected positions. Transforming the condition that adjacent selected positions have distance at least $k$ into an ordinary position-selection problem gives

$$
f(n,k)=\sum_{c\ge 0}\binom{n-(k-1)(c-1)}{c}
$$

The complexity is $O(n/k)$.

If there are many queries with the same $k$, use the first method; otherwise, calculate it using combinations.

### 1007

**Problem**: Given an $n \times m$ grid with obstacles in some cells, you can move up, down, left, or right from a cell. There are $k$ directed portals in the grid, and a cell may contain multiple portals. You may choose not to use a portal. There are $q$ queries, each asking whether it is possible to reach the destination from the starting point.

**Solution**: First divide all cells into connected components and assign an id to the connected component of each cell. Any two cells inside the same connected component are mutually reachable. Then consider the portals. A portal $(x_1,y_1) \to (x_2,y_2)$ is equivalent to adding an edge between the connected components containing these two cells. We only need to renumber the connected components and build a new graph, then run bfs from each connected component containing a portal to determine whether the connected components containing the starting point and destination are reachable.

### 1009

**Problem**: Given an array $P$ of length $2N$, where $P$ is a permutation of $2N$ and some elements are missing and represented by $0$. Define the value of the permutation as

$$
|P_1-P_2|+|P_3-P_4|+\cdots+|P_{2N-1}-P_{2N}|
$$

Find the number of ways to complete the permutation such that its value is maximized.

**Solution**: Divide the array into $N$ pairs of adjacent positions. If both numbers in a pair are non-zero, its contribution is already fixed, so we can ignore it. We only consider pairs containing at least one $0$. Suppose there are $k$ such pairs. Sort the corresponding $2k$ numbers and divide them into a lower half and an upper half. To maximize the contribution, each pair must contain one large number and one small number. Let $cnt$ be the number of pairs where both numbers are $0$. The answer is

$$
l!\times r!\times 2^{cnt}
$$

### 1012

**Problem**: Given an undirected graph with $n$ vertices and $m$ edges numbered $1,2,\dots,n$. There may be multiple edges connecting the same pair of vertices. The input graph is guaranteed to be bipartite. A cycle is a connected subgraph in which every vertex has degree $2$. If a cycle contains exactly $k$ vertices, it is called a $k$-cycle. The input guarantees that $k$ is prime. Find the number of different $k$-cycles in the graph.

**Solution**: Since the graph is bipartite, every cycle must have even length. Since $k$ is also prime, the cycle length can only be $2$. Therefore, we only need to count the number of ways to choose two different parallel edges between the same pair of vertices.
