"""
Value Iteration algorithm implementation for solving MDPs.

This module provides a reference implementation of the Value Iteration
algorithm for finding optimal policies in Markov Decision Processes.
"""

import numpy as np
from typing import Tuple, Dict, List, Any, Optional
import gymnasium as gym


def value_iteration(
    env: gym.Env,
    theta: float = 1e-6,
    gamma: float = 0.99,
    max_iterations: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Value Iteration algorithm for finding the optimal value function and policy.
    
    Args:
        env: The environment (must have discrete states and actions)
        theta: Convergence threshold
        gamma: Discount factor
        max_iterations: Maximum number of iterations
        
    Returns:
        Tuple of (optimal value function, optimal policy)
    """
    # Initialize value function
    V = np.zeros(env.nS)
    
    # Initialize policy
    policy = np.zeros([env.nS, env.nA])
    
    # Value iteration
    for i in range(max_iterations):
        delta = 0
        
        # Update each state
        for s in range(env.nS):
            v = V[s]
            
            # Calculate action values for this state
            action_values = np.zeros(env.nA)
            
            for a in range(env.nA):
                # Calculate expected value for each action
                for prob, next_state, reward, done in env.P[s][a]:
                    # Calculate expected value
                    action_values[a] += prob * (reward + gamma * V[next_state] * (not done))
            
            # Update value function with best action value
            best_action_value = np.max(action_values)
            delta = max(delta, np.abs(v - best_action_value))
            V[s] = best_action_value
        
        # Check convergence
        if delta < theta:
            print(f"Value iteration converged after {i+1} iterations.")
            break
    
    # Extract policy from value function
    for s in range(env.nS):
        action_values = np.zeros(env.nA)
        
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state] * (not done))
        
        # Greedy policy
        policy[s, np.argmax(action_values)] = 1.0
    
    return V, policy


if __name__ == "__main__":
    # Simple test with a small environment
    from environments.mars_rover import MarsRoverEnv
    
    # Create environment
    env = MarsRoverEnv(n_states=5)
    
    # Run value iteration
    V, policy = value_iteration(env)
    
    # Print results
    print("Optimal Value Function:")
    print(V)
    print("\nOptimal Policy:")
    for s in range(env.nS):
        a = np.argmax(policy[s])
        print(f"State {s}: {['LEFT', 'RIGHT'][a]}")
