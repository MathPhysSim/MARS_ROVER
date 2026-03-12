"""
Mars Rover Environment for Reinforcement Learning.

A customised Gymnasium environment that simulates a rover navigating a
one-dimensional Martian landscape with stochastic transitions.

Author: Simon Hirlaender
Course: Introduction to Reinforcement Learning 2026 — Paris Lodron University Salzburg
"""

from __future__ import annotations

import sys
from io import StringIO
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Mars Rover MDP
# ---------------------------------------------------------------------------

class MarsRoverEnv(gym.Env):
    """Discrete Mars Rover environment (Gymnasium v1.x compatible).

    The rover sits on a 1-D track of ``n_states`` non-terminal positions
    flanked by two terminal states (indices ``0`` and ``n_states + 1``).
    At every step the rover chooses LEFT (0) or RIGHT (1).  The actual
    transition is stochastic:

    * with probability ``p_forward`` the rover moves in the chosen direction,
    * with probability ``p_stay`` it stays in place,
    * with probability ``p_backward`` it moves in the opposite direction.

    Reaching a terminal state ends the episode and yields the corresponding
    reward.

    Attributes
    ----------
    LEFT, RIGHT : int
        Action constants.
    nS : int
        Total number of states (including terminals).
    nA : int
        Number of actions (always 2).
    P : dict
        Full transition table  ``P[s][a] -> [(prob, next_s, reward, done), ...]``.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    LEFT: int = 0
    RIGHT: int = 1

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        n_states: int = 5,
        p_stay: float = 1 / 3,
        p_backward: float = 1 / 3,
        left_side_reward: float = 1.0,
        right_side_reward: float = 10.0,
        render_mode: Optional[str] = None,
    ) -> None:
        """Create a Mars Rover environment.

        Parameters
        ----------
        n_states:
            Number of *non-terminal* states.
        p_stay:
            Probability of staying in place despite the chosen action.
        p_backward:
            Probability of moving opposite to the chosen action.
        left_side_reward:
            Reward for reaching the left terminal state (index 0).
        right_side_reward:
            Reward for reaching the right terminal state.
        render_mode:
            ``"human"`` prints to stdout; ``"ansi"`` returns a string.
        """
        super().__init__()

        # --- validate ------------------------------------------------
        if not (0 <= p_stay <= 1):
            raise ValueError(f"p_stay must be in [0, 1], got {p_stay}")
        if not (0 <= p_backward <= 1):
            raise ValueError(f"p_backward must be in [0, 1], got {p_backward}")
        if p_stay + p_backward > 1:
            raise ValueError(
                f"p_stay + p_backward must be ≤ 1, got {p_stay + p_backward}"
            )

        # --- store parameters ----------------------------------------
        self.n_states = n_states
        self.p_stay = p_stay
        self.p_backward = p_backward
        self.p_forward = 1.0 - p_stay - p_backward
        self.left_side_reward = left_side_reward
        self.right_side_reward = right_side_reward
        self.render_mode = render_mode

        # --- derived sizes -------------------------------------------
        self.nS: int = n_states + 2  # includes two terminal states
        self.nA: int = 2

        # --- spaces ---------------------------------------------------
        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        # --- transition table ----------------------------------------
        self.P = self._build_transitions()

        # --- initial state distribution (start in the middle) --------
        self.isd = np.zeros(self.nS, dtype=np.float64)
        self.isd[n_states // 2 + 1] = 1.0

        # --- internal state ------------------------------------------
        self.s: int = int(np.argmax(self.isd))

    # ------------------------------------------------------------------
    # Transition construction
    # ------------------------------------------------------------------

    def _build_transitions(self) -> dict[int, dict[int, list[tuple[float, int, float, bool]]]]:
        """Build the full transition table ``P[s][a]``."""
        P: dict[int, dict[int, list[tuple[float, int, float, bool]]]] = {
            s: {a: [] for a in range(self.nA)} for s in range(self.nS)
        }

        # --- terminal states (absorbing with reward) -----------------
        for a in range(self.nA):
            P[0][a] = [(1.0, 0, 0.0, True)]
            P[self.nS - 1][a] = [(1.0, self.nS - 1, 0.0, True)]

        # --- non-terminal states -------------------------------------
        for s in range(1, self.nS - 1):
            for a in (self.LEFT, self.RIGHT):
                P[s][a] = self._outcomes(s, a)

        return P

    def _outcomes(
        self, s: int, action: int
    ) -> list[tuple[float, int, float, bool]]:
        """Return ``[(prob, next_state, reward, terminated), ...]``."""
        move = -1 if action == self.LEFT else 1
        results: list[tuple[float, int, float, bool]] = []

        def _append(prob: float, next_s: int) -> None:
            next_s = int(np.clip(next_s, 0, self.nS - 1))
            is_terminal = next_s in (0, self.nS - 1)
            reward = {0: self.left_side_reward, self.nS - 1: self.right_side_reward}.get(next_s, 0.0)
            results.append((prob, next_s, reward, is_terminal))

        if self.p_forward > 0:
            _append(self.p_forward, s + move)
        if self.p_stay > 0:
            _append(self.p_stay, s)
        if self.p_backward > 0:
            _append(self.p_backward, s - move)

        return results

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[int, dict[str, Any]]:
        """Reset the environment and return ``(observation, info)``."""
        super().reset(seed=seed, options=options)
        self.s = int(self.np_random.choice(self.nS, p=self.isd))
        return self.s, {}

    def step(
        self, action: int
    ) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Execute *action* and return ``(obs, reward, terminated, truncated, info)``."""
        transitions = self.P[self.s][action]
        probs = [t[0] for t in transitions]
        idx = int(self.np_random.choice(len(transitions), p=probs))
        prob, next_state, reward, terminated = transitions[idx]
        self.s = next_state
        return next_state, float(reward), terminated, False, {"prob": prob}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> Optional[str]:
        """Render the current state of the environment."""
        desc = ["[T]"] + [f" {i} " for i in range(1, self.nS - 1)] + ["[T]"]
        desc[self.s] = f"<{desc[self.s].strip()}>"

        line = " ".join(desc)

        if self.render_mode == "ansi":
            outfile = StringIO()
            outfile.write(line + "\n")
            return outfile.getvalue()
        elif self.render_mode == "human":
            print(line)
        return None


# ---------------------------------------------------------------------------
# MRP Wrapper
# ---------------------------------------------------------------------------

class MarsRoverMRPWrapper(gym.Wrapper):
    """Convert the Mars Rover MDP into an MRP by fixing a policy.

    Parameters
    ----------
    env:
        A ``MarsRoverEnv`` instance.
    policy:
        Array of shape ``(nS, nA)`` where ``policy[s, a]`` is the
        probability of choosing action *a* in state *s*.
        Defaults to a uniform random policy.
    """

    def __init__(
        self,
        env: MarsRoverEnv,
        policy: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(env)
        nS: int = env.nS
        nA: int = env.nA

        if policy is None:
            self.policy = np.ones((nS, nA), dtype=np.float64) / nA
        else:
            if policy.shape != (nS, nA):
                raise ValueError(
                    f"Policy shape {policy.shape} does not match ({nS}, {nA})"
                )
            self.policy = np.asarray(policy, dtype=np.float64)

    def step(
        self, action: Optional[int] = None
    ) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Step using the fixed policy (the *action* argument is ignored)."""
        current_state: int = self.env.unwrapped.s  # type: ignore[attr-defined]
        action_probs = self.policy[current_state]
        chosen = int(self.np_random.choice(self.env.unwrapped.nA, p=action_probs))  # type: ignore[attr-defined]
        return self.env.step(chosen)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = MarsRoverEnv(render_mode="human")
    obs, info = env.reset(seed=42)

    print("Mars Rover Environment Demo")
    print("=" * 40)
    print(f"Observation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")
    print(f"Number of states  : {env.nS}")
    print(f"Initial state     : {obs}\n")

    for step_i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        action_name = "LEFT" if action == MarsRoverEnv.LEFT else "RIGHT"
        print(f"  Step {step_i + 1:>2}: {action_name:>5} → state={obs}, r={reward:.1f}, done={terminated}")
        env.render()

        if terminated or truncated:
            print("  ── episode reset ──")
            obs, info = env.reset()
