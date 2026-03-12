"""
Value Iteration for tabular MDPs.

Finds the optimal value function and deterministic policy by iteratively
applying the Bellman optimality operator.

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
class ConvergenceInfo:
    """Diagnostics returned alongside the solution."""

    iterations: int = 0
    converged: bool = False
    delta_history: list[float] = field(default_factory=list)


def value_iteration(
    env: gym.Env,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iterations: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, ConvergenceInfo]:
    """Run Value Iteration on an environment that exposes ``P``, ``nS``, ``nA``.

    Parameters
    ----------
    env:
        Discrete environment with attributes ``nS``, ``nA``, and ``P``.
    gamma:
        Discount factor.
    theta:
        Convergence threshold on the max Bellman residual.
    max_iterations:
        Upper bound on the number of sweeps.

    Returns
    -------
    V : np.ndarray
        Optimal state-value function of shape ``(nS,)``.
    policy : np.ndarray
        Deterministic policy of shape ``(nS,)`` with integer action indices.
    info : ConvergenceInfo
        Convergence diagnostics.
    """
    nS: int = env.nS  # type: ignore[attr-defined]
    nA: int = env.nA  # type: ignore[attr-defined]
    P = env.P  # type: ignore[attr-defined]

    V = np.zeros(nS, dtype=np.float64)
    info = ConvergenceInfo()

    for it in range(1, max_iterations + 1):
        delta = 0.0
        for s in range(nS):
            q = np.zeros(nA, dtype=np.float64)
            for a in range(nA):
                for prob, next_s, reward, done in P[s][a]:
                    q[a] += prob * (reward + gamma * V[next_s] * (1.0 - done))
            best = q.max()
            delta = max(delta, abs(V[s] - best))
            V[s] = best

        info.delta_history.append(delta)
        if delta < theta:
            info.converged = True
            info.iterations = it
            logger.info("Value Iteration converged after %d iterations (Δ=%.2e).", it, delta)
            break
    else:
        info.iterations = max_iterations
        logger.warning("Value Iteration did NOT converge within %d iterations (Δ=%.2e).", max_iterations, delta)

    # --- extract greedy policy ---
    policy = np.zeros(nS, dtype=np.intp)
    for s in range(nS):
        q = np.zeros(nA, dtype=np.float64)
        for a in range(nA):
            for prob, next_s, reward, done in P[s][a]:
                q[a] += prob * (reward + gamma * V[next_s] * (1.0 - done))
        policy[s] = int(q.argmax())

    return V, policy, info


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from environments.mars_rover import MarsRoverEnv

    env = MarsRoverEnv(n_states=5)
    V, policy, info = value_iteration(env)

    print(f"\nOptimal Value Function:  {np.array2string(V, precision=4)}")
    print(f"Optimal Policy:         {policy}")
    for s in range(env.nS):
        print(f"  State {s}: {'LEFT' if policy[s] == 0 else 'RIGHT'}")
    print(f"\nConverged: {info.converged} in {info.iterations} iterations")
