# Getting Started with Introduction to RL — 2026

This guide helps you set up your environment and start working with the course materials.

## Prerequisites

- **Python 3.10+** — [download](https://www.python.org/downloads/)
- **Git** — [download](https://git-scm.com/downloads)
- A code editor (VSCode, PyCharm, or Jupyter)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MathPhysSim/MARS_ROVER.git
cd MARS_ROVER
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
# Core install
pip install -e .

# With Jupyter notebook support
pip install -e ".[notebooks]"
```

### 4. Verify Installation

```bash
python -c "from environments.mars_rover import MarsRoverEnv; print('✓ OK')"
```

## First Steps

1. **Explore the environment** — Run `python environments/mars_rover/mars_rover_env.py` for a quick demo.
2. **Run Value Iteration** — Run `python algorithms/value_iteration.py`.
3. **Run Policy Iteration** — Run `python algorithms/policy_iteration.py`.
4. **Run Q-Learning** — Run `python algorithms/q_learning.py`.

## Resources

- [Gymnasium documentation](https://gymnasium.farama.org/)
- Sutton & Barto — *Reinforcement Learning: An Introduction*, Chapters 1–6
- Course lecture notes (available on Moodle)
