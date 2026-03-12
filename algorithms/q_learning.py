"""
Tabular Q-Learning with ε-greedy exploration.

A model-free, off-policy algorithm that directly learns the optimal
action-value function Q*(s, a) from sampled transitions.

Author: Simon Hirlaender, 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QLearningInfo:
    """Diagnostics returned alongside the Q-table."""

    total_episodes: int = 0
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)


def q_learning(
    env: gym.Env,
    n_episodes: int = 1_000,
    gamma: float = 0.99,
    alpha: float = 0.1,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    max_steps_per_episode: int = 200,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, QLearningInfo]:
    """Run tabular Q-learning on a discrete environment.

    Parameters
    ----------
    env:
        Gymnasium environment (discrete observation and action spaces).
    n_episodes:
        Number of training episodes.
    gamma:
        Discount factor.
    alpha:
        Learning rate.
    epsilon_start:
        Initial exploration rate.
    epsilon_end:
        Minimum exploration rate.
    epsilon_decay:
        Multiplicative decay applied to ε after each episode.
    max_steps_per_episode:
        Safety cap on episode length.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Q : np.ndarray
        Action-value table of shape ``(nS, nA)``.
    policy : np.ndarray
        Greedy policy of shape ``(nS,)`` derived from ``Q``.
    info : QLearningInfo
        Training diagnostics.
    """
    nS: int = env.observation_space.n  # type: ignore[union-attr]
    nA: int = env.action_space.n  # type: ignore[union-attr]

    Q = np.zeros((nS, nA), dtype=np.float64)
    info = QLearningInfo()
    epsilon = epsilon_start
    rng = np.random.default_rng(seed)

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=None)
        total_reward = 0.0
        steps = 0

        for _ in range(max_steps_per_episode):
            # ε-greedy action selection
            if rng.random() < epsilon:
                action = int(rng.integers(nA))
            else:
                action = int(Q[obs].argmax())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-update
            td_target = reward + gamma * Q[next_obs].max() * (1.0 - done)
            Q[obs, action] += alpha * (td_target - Q[obs, action])

            obs = next_obs
            total_reward += reward
            steps += 1

            if done:
                break

        # decay exploration
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        info.episode_rewards.append(total_reward)
        info.episode_lengths.append(steps)

        if ep % max(1, n_episodes // 10) == 0:
            avg_r = np.mean(info.episode_rewards[-100:])
            logger.info("Episode %5d | ε=%.3f | avg reward (last 100)=%.2f", ep, epsilon, avg_r)

    info.total_episodes = n_episodes

    # derive greedy policy
    policy = Q.argmax(axis=1).astype(np.intp)

    return Q, policy, info


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from environments.mars_rover import MarsRoverEnv

    env = MarsRoverEnv(n_states=5)
    Q, policy, info = q_learning(env, n_episodes=2_000, seed=42)

    print(f"\nQ-table:\n{np.array2string(Q, precision=3)}")
    print(f"\nLearned Policy: {policy}")
    for s in range(env.nS):
        print(f"  State {s}: {'LEFT' if policy[s] == 0 else 'RIGHT'}")
    print(f"\nFinal avg reward (last 100): {np.mean(info.episode_rewards[-100:]):.2f}")
