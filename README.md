# EAML-SSM: Experience-Augmented Meta-Learning with State-Space Models

[![CI](https://github.com/sunghunkwag/EAML-SSM/actions/workflows/ci.yml/badge.svg)](https://github.com/sunghunkwag/EAML-SSM/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid research framework that combines high-performance meta-learning with experience-augmented test-time adaptation.

This repository merges two key concepts:
1.  **High-Performance Meta-RL (`ssm-metarl-testcompute`)**: A scalable PyTorch-based framework using State-Space Models (SSMs) and Model-Agnostic Meta-Learning (MAML) for stateful policy adaptation.
2.  **Experience-Based Reasoning (`lownocompute-ai-baseline`)**: A novel architecture that uses an `ExperienceBuffer` to store and retrieve past experiences, enabling more robust adaptation in new environments.

The result is a system that acquires a fast meta-learned initialization (via MAML) and then performs robust, experience-augmented adaptation at test-time by leveraging both current task data and relevant past experiences.

## 🎯 Core Idea: Hybrid Adaptation

Traditional Test-Time Adaptation (TTA) adapts a model using *only* the data from the new task. This can be unstable if the new data is scarce or non-representative.

**EAML-SSM** implements a hybrid approach:

1.  **Meta-Training (MAML)**: The SSM model is meta-trained to be quickly adaptable to new tasks. During this phase, all encountered data is stored in the `ExperienceBuffer`.
2.  **Test-Time Adaptation (Adapter)**: When faced with a new task, the `Adapter` module performs gradient updates using a combined loss:
    * `loss_current`: Loss on the *new task's* data.
    * `loss_experience`: Loss on a batch of *past experiences* sampled from the buffer.
    * `total_loss = loss_current + (experience_weight * loss_experience)`

This approach "grounds" the adaptation process, preventing the model from overfitting to the few-shot data of the new task and improving stability.

## 🏗️ Architecture
