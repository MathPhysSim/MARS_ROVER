"""
Mars Rover Environment for Reinforcement Learning.

This module provides a customized Gymnasium environment simulating
a Mars rover navigating across a linear landscape.
"""

import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text import DiscreteEnv
from typing import Optional, Tuple, Dict, List, Any, Union


class MarsRoverEnv(DiscreteEnv):
    """
    Mars Rover environment for reinforcement learning.
    
    The rover navigates a linear path with discrete states and
    probabilistic transitions. The goal is to reach the right terminal state.
    
    Attributes:
        LEFT (int): Constant representing the left movement action (0)
        RIGHT (int): Constant representing the right movement action (1)
    """
    
    LEFT = 0
    RIGHT = 1
    
    def __init__(
        self, 
        n_states: int = 5, 
        p_stay: float = 0.2, 
        p_backward: float = 0.1,
        left_side_reward: float = 0.0,
        right_side_reward: float = 1.0
    ):
        """
        Initialize the Mars Rover environment.
        
        Args:
            n_states: Number of non-terminal states (excluding terminal states)
            p_stay: Probability of staying in the same state despite taking an action
            p_backward: Probability of moving in the opposite direction of the action
            left_side_reward: Reward for reaching the left terminal state
            right_side_reward: Reward for reaching the right terminal state
        """
        # Validate parameters
        assert 0 <= p_stay <= 1, "p_stay must be between 0 and 1"
        assert 0 <= p_backward <= 1, "p_backward must be between 0 and 1"
        assert p_stay + p_backward <= 1, "Sum of p_stay and p_backward must not exceed 1"
        
        self.n_states = n_states
        self.p_stay = p_stay
        self.p_backward = p_backward
        self.left_side_reward = left_side_reward
        self.right_side_reward = right_side_reward
        
        # Total states including terminal states
        nS = n_states + 2
        # Actions: left and right
        nA = 2
        
        # Initial state distribution: always start in middle state
        isd = np.zeros(nS)
        isd[n_states // 2 + 1] = 1.0
        
        # Create transition probability matrix
        P = self._create_transition_matrix(nS, nA)
        
        # Initialize the DiscreteEnv superclass
        super(MarsRoverEnv, self).__init__(nS, nA, P, isd)
    
    def _create_transition_matrix(self, nS: int, nA: int) -> Dict:
        """
        Create the transition probability matrix.
        
        Args:
            nS: Total number of states including terminal states
            nA: Number of actions
            
        Returns:
            Dict: Transition probability matrix
        """
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        
        # Terminal states
        P[0][self.LEFT] = [(1.0, 0, self.left_side_reward, True)]
        P[0][self.RIGHT] = [(1.0, 0, self.left_side_reward, True)]
        
        P[nS-1][self.LEFT] = [(1.0, nS-1, self.right_side_reward, True)]
        P[nS-1][self.RIGHT] = [(1.0, nS-1, self.right_side_reward, True)]
        
        # Non-terminal states
        p_forward = 1.0 - self.p_stay - self.p_backward
        
        for s in range(1, nS-1):
            # Left action
            P[s][self.LEFT] = self._calculate_outcomes(
                s, self.LEFT, p_forward, nS
            )
            
            # Right action
            P[s][self.RIGHT] = self._calculate_outcomes(
                s, self.RIGHT, p_forward, nS
            )
        
        return P
    
    def _calculate_outcomes(self, state: int, action: int, p_forward: float, nS: int) -> List[Tuple]:
        """
        Calculate possible outcomes for a given state-action pair.
        
        Args:
            state: Current state
            action: Chosen action
            p_forward: Probability of moving in the intended direction
            nS: Total number of states
            
        Returns:
            List of (probability, next_state, reward, done) tuples
        """
        outcomes = []
        
        # Stay in the same state
        if self.p_stay > 0:
            outcomes.append((self.p_stay, state, 0.0, False))
        
        # Move backward
        if self.p_backward > 0:
            next_state = state + 1 if action == self.LEFT else state - 1
            
            # Check if moving backward leads to a terminal state
            if next_state == 0:
                outcomes.append((self.p_backward, next_state, self.left_side_reward, True))
            elif next_state == nS - 1:
                outcomes.append((self.p_backward, next_state, self.right_side_reward, True))
            else:
                outcomes.append((self.p_backward, next_state, 0.0, False))
        
        # Move forward
        if p_forward > 0:
            next_state = state - 1 if action == self.LEFT else state + 1
            
            # Check if moving forward leads to a terminal state
            if next_state == 0:
                outcomes.append((p_forward, next_state, self.left_side_reward, True))
            elif next_state == nS - 1:
                outcomes.append((p_forward, next_state, self.right_side_reward, True))
            else:
                outcomes.append((p_forward, next_state, 0.0, False))
        
        return outcomes
    
    def render(self, mode: str = "human") -> None:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        outfile = StringIO() if mode == 'ansi' else None
        
        desc = ["T"] + ["O"] * (self.n_states) + ["T"]
        desc[self.s] = "X"
        
        if outfile is not None:
            outfile.write(" ".join(desc) + "\n")
            return outfile
        else:
            print(" ".join(desc))


class MarsRoverMRPWrapper(gym.Wrapper):
    """
    A wrapper that converts the MDP into an MRP by fixing a policy.
    
    This wrapper demonstrates how to convert a Markov Decision Process (MDP)
    into a Markov Reward Process (MRP) by following a fixed policy.
    
    Attributes:
        env: The wrapped environment
        policy: Fixed policy to follow (probability distribution over actions)
    """
    
    def __init__(self, env: MarsRoverEnv, policy: Optional[np.ndarray] = None):
        """
        Initialize the MRP wrapper.
        
        Args:
            env: Mars Rover environment to wrap
            policy: Fixed policy to follow, must be a 2D array of shape (nS, nA)
                   where policy[s,a] is the probability of taking action a in state s.
                   If None, a uniform random policy will be used.
        """
        super(MarsRoverMRPWrapper, self).__init__(env)
        
        if policy is None:
            # Default to uniform random policy
            self.policy = np.ones((env.nS, env.nA)) / env.nA
        else:
            assert policy.shape == (env.nS, env.nA), "Policy shape must match environment dimensions"
            self.policy = policy
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.
        
        Returns:
            Tuple of observation and info dict
        """
        return self.env.reset(**kwargs)
    
    def step(self, action: Optional[int] = None) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment following the fixed policy.
        
        The action parameter is ignored since we follow the fixed policy.
        
        Args:
            action: Ignored parameter
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Sample action from policy for current state
        current_state = self.env.s
        action_probs = self.policy[current_state]
        action = np.random.choice(self.env.nA, p=action_probs)
        
        # Take the sampled action in the environment
        return self.env.step(action)


# Add this import if using StringIO for rendering
from io import StringIO

# If this file is run as a script, provide a simple demo
if __name__ == "__main__":
    env = MarsRoverEnv()
    obs = env.reset()[0]
    
    print("Mars Rover Environment Demo")
    print("---------------------------")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial state: {obs}")
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, State={obs}, Reward={reward}, Done={terminated or truncated}")
        env.render()
        
        if terminated or truncated:
            print("Environment reset!")
            obs = env.reset()[0]
