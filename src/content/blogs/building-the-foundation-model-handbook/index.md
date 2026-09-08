---
title: 'Building The Foundation Model Handbook'
publishDate: 2026-09-06
updatedDate: 2026-09-08
category: technical
tags: [Foundation Models Handbook, LLM, Project]
language: en
description: 'Presenting Version 1.0 of The Foundation Model Handbook, a 39-chapter technical handbook on modern LLMs and their systems.'
heroImage:
  src: /assets/img/posts/foundation-model-handbook/handbook-card-cover-pink.png
  alt: 'Pink moonlit lake with lanterns and a figure in traditional dress'
  color: '#aa6d8e'
---

I have completed Version 1.0 of a long-term project called [**The Foundation Model Handbook**](https://github.com/pig7selene/the-foundation-model-handbook). The first complete edition is a 39-chapter technical monograph on modern foundation models and LLM systems. It remains a structured record of what I have learned rather than an attempt to be an authoritative textbook, but the core roadmap is now complete and available as a single assembled book.

> Read the [complete Version 1.0 PDF](https://github.com/pig7selene/the-foundation-model-handbook/raw/refs/heads/main/build/the-foundation-model-handbook-v1.0.pdf), or browse the [source and standalone chapter PDFs on GitHub](https://github.com/pig7selene/the-foundation-model-handbook).

## Why I Started It

While learning about foundation models, I found that the relevant knowledge was scattered across courses, papers, technical reports, documentation, and my own notes. Each source explained one part well, but the connections between the parts were easy to lose. Tokenization, Transformer architecture, pretraining, alignment methods, and inference systems often appeared in different places and at very different levels of detail.

I wanted a single place where I could organize what I had actually learned into a coherent structure. The project began as a collection of notes. As the notes grew, I gave them a shared notation system, a consistent chapter structure, a bibliography, and a common Typst layout. That gradual change turned the notes into a handbook; completing the full reading path turned the handbook into a finished first edition.

## A Learning Path Through Foundation Models

Version 1.0 contains 39 independently readable chapters across seven parts. The numbering is also the intended reading order, moving monotonically from Chapter 1 to Chapter 39:

1. **Foundations** introduces tokenization and the discrete interface between text and the model.
2. **Architecture** develops the decoder-only Transformer, Attention and Position Encoding, feed-forward networks, normalization, and residual connections.
3. **Pretraining** connects the language-modeling objective to data, optimization, numerical stability, scaling laws, distributed training, evaluation, checkpointing, and practical FSDP state.
4. **Post-training, Alignment, and Adaptation** covers SFT, parameter-efficient fine-tuning, preference data, Reward Modeling, RLHF / PPO, DPO, GRPO, Reasoning RL, evaluation, online policy improvement, and distributed RL systems.
5. **Inference and Serving** follows a trained model through autoregressive execution, KV Cache management, Quantization, batching, scheduling, speculative decoding, distributed inference, and end-to-end performance optimization.
6. **Efficient Attention and Long Context** studies FlashAttention, efficient head-representation designs, sparse attention, and long-context trade-offs.
7. **Retrieval-Augmented Generation and Knowledge Augmentation** develops the complete retrieval pipeline from embeddings, vector search, chunking, and hybrid retrieval to reranking, advanced RAG architectures, evaluation, and diagnosis.

The sequence begins with the interface between text and the model. Tokenization leads into the decoder-only Transformer and the components that define its computation. Those chapters establish the notation, tensor shapes, and architectural contracts assumed by everything that follows.

The next part asks how that architecture becomes a trained model. It moves from the language-modeling objective and Pretraining Data into Optimization, numerical stability, Scaling Laws, Distributed Training, diagnostics, checkpointing, and FSDP. I find this progression useful because it connects the mathematical objective to the data and systems work required to optimize it at scale.

Post-training then changes the question from how a model learns a general next-token distribution to how its behavior is adapted. SFT provides the first step, followed by parameter-efficient adaptation, preference data and Reward Modeling, RLHF / PPO, DPO, GRPO, Reasoning RL, evaluation, iterative on-policy improvement, and distributed RL execution. Keeping these methods in one sequence makes their assumptions and trade-offs easier to compare: what data they require, whether learning is online or offline, how rewards enter the objective, and where instability can appear.

Inference & Serving follows the trained model into actual execution and connects model structure to latency, throughput, memory capacity, and the behavior of a serving system under real request workloads. Efficient Attention and Long Context then extend that systems view, before the final RAG part shows how a model can retrieve, refine, use, and evaluate external knowledge.

This is the learning path I wanted when I started studying the subject. It can be used to learn foundation models systematically, prepare background knowledge for LLM or foundation-model algorithm internships, supplement university coursework, revisit an important concept, or explore systems topics that do not fit neatly into a single course.

![A page from a standalone chapter in The Foundation Model Handbook](/assets/img/posts/foundation-model-handbook/chapter-preview.png)

## How the Handbook Is Built

Each chapter is written as a structured technical note and is intended to be readable on its own. Mathematical notation and terminology are kept consistent across chapters, while established professional terms remain in English. Consequential technical claims are traced back to primary papers, official reports, or standard references rather than being left as unsupported summaries.

The chapters are typeset with Typst. Shared style, notation, and chapter conventions keep the book consistent, and every chapter can be compiled into a standalone PDF. The root `main.typ` assembles those chapters with book-level front matter, a table of contents, parts, running headers, and page numbering. The repository now publishes both the standalone chapter PDFs and the complete Version 1.0 release PDF.

AI and Codex assist with parts of the workflow, including organizing material, checking terminology and notation, researching references, formatting, compiling, visually inspecting pages, and maintaining the repository. I still decide the scope of each chapter, work through the sources, connect the ideas, and take responsibility for the final text. I think of it as an AI-assisted learning and writing project, not an autonomously generated handbook.

## What Version 1.0 Completes

Version 1.0 completes the core roadmap I originally wanted to study: LLM Architecture, Pretraining, Post-training and Alignment, Parameter-Efficient Fine-Tuning, Inference Optimization, and LLM Systems. Efficient Attention, Long Context, and Retrieval-Augmented Generation provide the prerequisite and extension material needed to connect those areas into one coherent reading path.

The project will still be maintained: references can be improved, explanations can be clarified, and mistakes can be corrected. Any genuinely new subject, however, will belong to a later edition rather than being silently inserted into the Version 1.0 roadmap. A future chapter should meet the same requirements as the current ones: explicit prerequisites, source-backed explanation, consistent notation, successful compilation, and visual review.

The complete edition is available as [The Foundation Model Handbook Version 1.0 PDF](https://github.com/pig7selene/the-foundation-model-handbook/raw/refs/heads/main/build/the-foundation-model-handbook-v1.0.pdf). The source, standalone chapters, bibliography, templates, and build scripts are available at [github.com/pig7selene/the-foundation-model-handbook](https://github.com/pig7selene/the-foundation-model-handbook).
