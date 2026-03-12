"""
Policy Iteration for tabular MDPs.

Alternates between *policy evaluation* (solving the Bellman expectation
equation for a fixed policy) and *policy improvement* (greedy update)
until the policy stabilises.

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
class PIConvergenceInfo:
    """Diagnostics returned alongside the solution."""

    policy_iterations: int = 0
    eval_iterations_per_step: list[int] = field(default_factory=list)
    converged: bool = False


def _policy_evaluation(
    policy: np.ndarray,
    P: dict,
    nS: int,
    nA: int,
    gamma: float,
    theta: float,
    max_eval_iterations: int,
) -> tuple[np.ndarray, int]:
    """Evaluate a deterministic *policy* and return ``(V, n_iterations)``."""
    V = np.zeros(nS, dtype=np.float64)
    for it in range(1, max_eval_iterations + 1):
        delta = 0.0
        for s in range(nS):
            a = int(policy[s])
            v_new = 0.0
            for prob, next_s, reward, done in P[s][a]:
                v_new += prob * (reward + gamma * V[next_s] * (1.0 - done))
            delta = max(delta, abs(V[s] - v_new))
            V[s] = v_new
        if delta < theta:
            return V, it
    return V, max_eval_iterations


def policy_iteration(
    env: gym.Env,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iterations: int = 1_000,
    max_eval_iterations: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, PIConvergenceInfo]:
    """Run Policy Iteration on a discrete environment.

    Parameters
    ----------
    env:
        Environment with ``nS``, ``nA``, ``P`` attributes.
    gamma:
        Discount factor.
    theta:
        Convergence threshold for policy evaluation.
    max_iterations:
        Maximum policy-improvement sweeps.
    max_eval_iterations:
        Maximum iterations per policy-evaluation call.

    Returns
    -------
    V : np.ndarray
        State-value function of shape ``(nS,)``.
    policy : np.ndarray
        Deterministic policy of shape ``(nS,)`` with integer actions.
    info : PIConvergenceInfo
        Diagnostics.
    """
    nS: int = env.nS  # type: ignore[attr-defined]
    nA: int = env.nA  # type: ignore[attr-defined]
    P = env.P  # type: ignore[attr-defined]

    # start with a random deterministic policy
    policy = np.zeros(nS, dtype=np.intp)
    info = PIConvergenceInfo()

    for pi_step in range(1, max_iterations + 1):
        # --- evaluation ---
        V, n_eval = _policy_evaluation(policy, P, nS, nA, gamma, theta, max_eval_iterations)
        info.eval_iterations_per_step.append(n_eval)

        # --- improvement ---
        stable = True
        for s in range(nS):
            q = np.zeros(nA, dtype=np.float64)
            for a in range(nA):
                for prob, next_s, reward, done in P[s][a]:
                    q[a] += prob * (reward + gamma * V[next_s] * (1.0 - done))
            best_a = int(q.argmax())
            if best_a != policy[s]:
                stable = False
                policy[s] = best_a

        if stable:
            info.converged = True
            info.policy_iterations = pi_step
            logger.info("Policy Iteration converged after %d steps.", pi_step)
            break
    else:
        info.policy_iterations = max_iterations
        logger.warning("Policy Iteration did NOT converge within %d steps.", max_iterations)

    return V, policy, info


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from environments.mars_rover import MarsRoverEnv

    env = MarsRoverEnv(n_states=5)
    V, policy, info = policy_iteration(env)

    print(f"\nOptimal Value Function:  {np.array2string(V, precision=4)}")
    print(f"Optimal Policy:         {policy}")
    for s in range(env.nS):
        print(f"  State {s}: {'LEFT' if policy[s] == 0 else 'RIGHT'}")
    print(f"\nConverged: {info.converged} in {info.policy_iterations} policy steps")
    print(f"Eval iterations per step: {info.eval_iterations_per_step}")
