# Introduction to Reinforcement Learning — 2026

[![Python ≥3.10](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.x-green.svg)](https://gymnasium.farama.org/)

Course materials for the **Introduction to Reinforcement Learning** lecture and exercises held at Paris Lodron University Salzburg in **2026** by **Simon Hirlaender**.

<img width="500" alt="Mars Rover" src="https://github.com/MathPhysSim/Introduction2RL_2024/assets/22523245/419f30be-12f0-4445-a077-56b0c8f03eda">

## 📚 Course Overview

This repository provides:

- A fully self-contained **Mars Rover** Gymnasium environment
- Reference implementations of classical RL algorithms
  - Value Iteration
  - Policy Iteration
  - Tabular Q-Learning
- Exercise materials and documentation

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Knowledge of basic Python programming
- Familiarity with probability and linear algebra

### Installation

```bash
# Clone
git clone https://github.com/MathPhysSim/MARS_ROVER.git
cd MARS_ROVER

# (Optional) virtual environment
python -m venv .venv && source .venv/bin/activate

# Install (editable mode)
pip install -e .

# With notebook support
pip install -e ".[notebooks]"
```

## 📂 Project Structure

```
.
├── algorithms/
│   ├── __init__.py
│   ├── value_iteration.py     # Value Iteration (tabular)
│   ├── policy_iteration.py    # Policy Iteration (tabular)
│   └── q_learning.py          # Tabular Q-Learning (ε-greedy)
├── environments/
│   └── mars_rover/
│       ├── __init__.py
│       └── mars_rover_env.py  # MarsRoverEnv + MRP wrapper
├── docs/
│   └── getting_started.md
├── exercises/
│   └── week1/
│       └── README.md
├── notebooks/                  # (add your own Jupyter notebooks here)
├── pyproject.toml
└── README.md
```

## 🎮 Quick Example

```python
from environments.mars_rover import MarsRoverEnv
from algorithms import value_iteration

# Create environment
env = MarsRoverEnv(n_states=5)

# Solve with Value Iteration
V, policy, info = value_iteration(env, gamma=0.99)
print(f"Converged in {info.iterations} iterations")
print(f"Value function: {V}")
print(f"Policy: {policy}")  # 0 = LEFT, 1 = RIGHT
```

## 📄 License

MIT — see individual files for details.

## 👤 Author

**Simon Hirlaender**
Paris Lodron University Salzburg — 2026
