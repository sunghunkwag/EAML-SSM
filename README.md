# EAML-SSM: Experience-Augmented Meta-Learning with State-Space Models

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A hybrid research framework that combines high-performance meta-learning with experience-augmented test-time adaptation.

This repository merges two key concepts:

1. **High-Performance Meta-RL (`ssm-metarl-testcompute`)**: A scalable PyTorch-based framework using State-Space Models (SSMs) and Model-Agnostic Meta-Learning (MAML) for stateful policy adaptation.
2. **Experience-Based Reasoning (`lownocompute-ai-baseline`)**: A novel architecture that uses an `ExperienceBuffer` to store and retrieve past experiences, enabling more robust adaptation in new environments.

The result is a system that acquires a fast meta-learned initialization (via MAML) and then performs robust, experience-augmented adaptation at test-time by leveraging both current task data and relevant past experiences.

## 🎯 Core Idea: Hybrid Adaptation

Traditional Test-Time Adaptation (TTA) adapts a model using *only* the data from the new task. This can be unstable if the new data is scarce or non-representative.

**EAML-SSM** implements a hybrid approach:

1. **Meta-Training (MAML)**: The SSM model is meta-trained to be quickly adaptable to new tasks. During this phase, all encountered data is stored in the `ExperienceBuffer`.
2. **Test-Time Adaptation (Adapter)**: When faced with a new task, the `Adapter` module performs gradient updates using a combined loss:
   - `loss_current`: Loss on the *new task's* data.
   - `loss_experience`: Loss on a batch of *past experiences* sampled from the buffer.
   - `total_loss = loss_current + (experience_weight * loss_experience)`

This approach "grounds" the adaptation process, preventing the model from overfitting to the few-shot data of the new task and improving stability.

## 🏗️ Architecture

```
                   ┌───────────────────┐
                   │  Meta-Training    │
                   │  (MAML)           │
                   └─────────┬─────────┘
                             │ (meta-learned model)
                             │
 ┌───────────────────┐       │       ┌───────────────────┐
 │ collect_data()    ├───────┼───────►  ExperienceBuffer │
 └─────────┬─────────┘       │       │  (Stores all past │
           │ (new data)      │       │   experiences)    │
           │                 │       └─────────┬─────────┘
           │                 │                 │ (sampled experiences)
           │                 ▼                 │
           │   ┌───────────────────────────┐   │
           └───►   Adapter (Test-Time)     ◄───┘
               │ 1. Get new data (x, y)    │
               │ 2. Get experience (x_exp) │
               │ 3. Compute combined_loss  │
               │ 4. optimizer.step()       │
               └───────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/sunghunkwag/EAML-SSM.git
cd EAML-SSM

# Install dependencies (including torch and gymnasium)
pip install -e .

# For development (to run tests)
pip install -e .[dev]
```

### Run Main Demo

This will run the full pipeline: meta-training, experience accumulation, and the hybrid test-time adaptation demo.

```bash
python main.py --env_name CartPole-v1 --num_epochs 50
```

### Expected Output

```
...
Epoch 40, Meta Loss: 0.0123
Epoch 45, Meta Loss: 0.0119
MetaMAML training completed.
Experience Buffer populated with 50000 experiences.

Starting test-time adaptation with Experience Buffer...
...
Adaptation step 0, Hybrid Loss: 0.0456, Steps taken: 5
Adaptation step 10, Hybrid Loss: 0.0231, Steps taken: 5
...
Adaptation step 40, Hybrid Loss: 0.0102, Steps taken: 5
Adaptation step 45, Hybrid Loss: 0.0098, Steps taken: 5
Adaptation completed.

=== Hybrid Execution completed successfully ===
```

## 🛠️ Core Components

- `core/ssm.py`: PyTorch-based State-Space Model.
- `meta_rl/meta_maml.py`: MAML implementation for stateful models.
- `experience/experience_buffer.py`: **[New]** PyTorch-based circular buffer for storing and sampling `(observation, target)` tensors.
- `adaptation/test_time_adaptation.py`: **[Modified]** `Adapter` class that now accepts the `ExperienceBuffer` and performs hybrid-loss updates.
- `main.py`: **[Modified]** Main script that orchestrates the entire process, including buffer initialization, population, and injection into the `Adapter`.

## 🧪 Running Tests

```bash
pytest -v
```

This will run all unit tests, including new tests for the `ExperienceBuffer` and the modified `Adapter`.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines . 

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{eaml_ssm_2025,
  title={EAML-SSM: Experience-Augmented Meta-Learning with State-Space Models},
  author={Kwag, Sunghun},
  year={2025},
  url={https://github.com/sunghunkwag/EAML-SSM}
}
```

