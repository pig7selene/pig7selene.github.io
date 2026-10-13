---
title: "算法竞赛每日一题"
publishDate: 2026-10-13
category: daily
tags: [xcpc]
language: zh
description: "算法竞赛每日一题及个人题目解析"
heroImage:
  src: /assets/img/posts/sj-daily/daily-problem-cover.png
  alt: "Two illustrated characters against a pink and purple backdrop"
  color: "#a863b7"
---

这里会不定期更新做到的有意思的算法题，仅更新个人的思路，代码实现就不贴了～

## 2026.10.13
[CF 2275G](https://codeforces.com/contest/2275/problem/G)

比较水的一题，可以快速过一下，首先将原有边按价值升序排序，用 DSU 维护连通性，两端不连通就保留并合并，否则卖掉，记卖边收入为 $A$、连通块数量为 $c$，此时至少需要新增 $c−1$ 条电缆。然后将保留边按价值降序排列为 $w₁、w₂、\cdots$，考虑继续拆边重建：拆掉第 $i$ 条边的收入为 $wᵢ$，额外铺设费用为 $(c−1+i)x$，因此仅当 $wᵢ > (c−1+i)x$ 时值得拆。由于边的价值不增、额外费用递增，可以对每个 $x$ 二分找到值得拆掉的边数 $k$。预处理保留边的前缀和 $Sₖ$，最终卖边收入为 $A+Sₖ$，新增电缆数量为 $c−1+k$，利润为 $A+Sₖ−x(c−1+k)(c+k)/2$。时间复杂度为 $O(m log m + q log n)$
