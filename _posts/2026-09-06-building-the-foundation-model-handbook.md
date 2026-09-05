---
title: "Building The Foundation Model Handbook"
date: 2026-09-06 01:00:00 +0800
categories: [Project]
tags: [Foundation Models Handbook, LLM, Project]
description: "Introducing The Foundation Model Handbook, a long-term set of structured notes for learning modern foundation models."
card_image:
  path: /assets/img/posts/foundation-model-handbook/handbook-card-cover-pink.png
  alt: "Pink moonlit lake with lanterns and a figure in traditional dress"
---

I have been working on a long-term project called [**The Foundation Model Handbook**](https://github.com/pig7selene/the-foundation-model-handbook). It is not a finished textbook or an attempt to be an authoritative reference. It is a structured record of what I have learned about modern foundation models, written so that I can return to it, revise it, and keep extending it over time.

> Read [The Foundation Model Handbook on GitHub](https://github.com/pig7selene/the-foundation-model-handbook).
{: .chinese-version-link }

## Why I Started It

While learning about foundation models, I found that the relevant knowledge was scattered across courses, papers, technical reports, documentation, and my own notes. Each source explained one part well, but the connections between the parts were easy to lose. Tokenization, Transformer architecture, pretraining, alignment methods, and inference systems often appeared in different places and at very different levels of detail.

I wanted a single place where I could gradually organize what I had actually learned into a coherent structure. The project began as a collection of notes. As the notes grew, I started giving them a shared notation system, a consistent chapter structure, a bibliography, and a common Typst layout. That gradual change is what turned the notes into a handbook.

## A Learning Path Through Foundation Models

At the time of writing, the repository contains 22 standalone chapters across five parts: Foundations, Transformer Architecture, Pretraining, Post-training, and Inference & Serving.

The sequence begins with the interface between text and the model. Tokenization leads into the decoder-only Transformer, Attention and Position Encoding, feed-forward networks, normalization, and residual connections. These topics establish the computation that later chapters assume.

The next part asks how that architecture becomes a trained model. It moves through the language-modeling objective and Pretraining Data, then into Optimization, numerical stability, Scaling Laws, Distributed Training, and training diagnostics. I find this progression useful because it connects the mathematical objective to the data and systems work required to optimize it at scale.

Post-training then changes the question from how a model learns a general next-token distribution to how its behavior is adapted. SFT provides the first step, followed by preference data and Reward Modeling, RLHF / PPO, DPO, GRPO, and Reasoning RL. Keeping these methods in one sequence makes their assumptions and trade-offs easier to compare: what data they require, whether learning is online or offline, how rewards enter the objective, and where instability can appear.

Finally, Inference & Serving follows the trained model into actual execution. The current chapters cover autoregressive generation, KV Cache and memory optimization, Quantization, and the batching and scheduling decisions behind modern LLM inference. This part connects model structure to latency, throughput, memory capacity, and the behavior of a serving system under real request workloads.

This is the learning path I wanted when I started studying the subject. It can be used to learn foundation models systematically, prepare background knowledge for LLM or foundation-model algorithm internships, supplement university coursework, revisit an important concept, or explore systems topics that do not fit neatly into a single course.

![A page from a standalone chapter in The Foundation Model Handbook](/assets/img/posts/foundation-model-handbook/chapter-preview.png){: width="620" }

## How the Handbook Is Built

Each chapter is written as a structured technical note and is intended to be readable on its own. Mathematical notation and terminology are kept consistent across chapters, while established professional terms remain in English. Consequential technical claims are traced back to primary papers, official reports, or standard references rather than being left as unsupported summaries.

The chapters are typeset with Typst. Shared style, notation, and chapter conventions keep later additions consistent, and every chapter can be compiled into a standalone PDF. The implementation is deliberately simple: the layout should support reading and revision without becoming the focus of the project.

AI and Codex assist with parts of the workflow, including organizing material, checking terminology and notation, researching references, formatting, compiling, visually inspecting pages, and maintaining the repository. I still decide the scope of each chapter, work through the sources, connect the ideas, and take responsibility for the final text. I think of it as an AI-assisted learning and writing project, not an autonomously generated handbook.

## Where It Goes Next

The immediate plan is to continue Inference & Serving with attention kernels, speculative decoding, distributed serving, and more detailed latency-throughput optimization. The repository roadmap also includes Retrieval-Augmented Generation, Agents and Tool Use, and Multimodal Foundation Models.

These are directions rather than promises of completeness. A new part will appear only when I have studied its first topic carefully enough to write and review a chapter. For now, the goal is simply to keep a sustainable rhythm: learn one subject, turn it into a clear chapter, and connect it to what is already there.

The project is available at [github.com/pig7selene/the-foundation-model-handbook](https://github.com/pig7selene/the-foundation-model-handbook).
