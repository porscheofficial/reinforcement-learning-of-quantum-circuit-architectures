"""
This module defines the ReplayBuffer class, which is a crucial component for
off-policy reinforcement learning algorithms like Soft Actor-Critic (SAC).

The replay buffer stores transitions (state, action, reward, next_state)
observed by the agent, allowing it to sample and learn from a diverse set of
past experiences.
"""
from typing import Any, Tuple

import numpy as np
import torch

from QuantumStateEnv import QuantumStateEnv


class ReplayBuffer:
    """
    A circular buffer to store and sample transitions for off-policy RL.
    """

    def __init__(self, cfg: Any, device: torch.device, env: QuantumStateEnv):
        """
        Initializes the ReplayBuffer.

        Args:
            cfg: A configuration object with replay buffer parameters.
            device: The torch device (CPU or CUDA) to store the buffer on.
            env: The environment instance to get state and action dimensions.
        """
        self.device = device
        self.capacity = cfg.ReplayBuffer["capacity"]
        
        # Pre-allocate memory for the buffer for efficiency
        self.states = torch.zeros((self.capacity, env.state_size), dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.capacity, 1), dtype=torch.int64, device=device)
        self.continuous_actions = torch.zeros((self.capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((self.capacity, env.state_size), dtype=torch.float32, device=device)
        self.energies = torch.zeros((self.capacity, 1), dtype=torch.float32, device=device)
        self.energies_before = torch.zeros((self.capacity, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((self.capacity, 1), dtype=torch.bool, device=device)
        self.indices = torch.zeros((self.capacity, 1), dtype=torch.int64, device=device)
        
        self.current_index = 0
        self.size = 0

    def add_transition(self, state: np.ndarray, action: int, continuous_action: float,
                       next_state: np.ndarray, energy: float, energy_before: float,
                       done: bool, index: int):
        """
        Adds a new transition to the buffer.

        If the buffer is full, the oldest transition is overwritten.
        """
        idx = self.current_index
        
        self.states[idx] = torch.from_numpy(state).to(self.device)
        self.actions[idx] = action
        self.continuous_actions[idx] = continuous_action
        self.next_states[idx] = torch.from_numpy(next_state).to(self.device)
        self.energies[idx] = energy
        self.energies_before[idx] = energy_before
        self.dones[idx] = done
        self.indices[idx] = index
        
        # Update the index and buffer size
        self.current_index = (self.current_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Samples a random batch of transitions from the buffer.

        Args:
            batch_size: The number of transitions to sample.

        Returns:
            A tuple of tensors, each containing a batch of a specific transition component.
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough samples in buffer to draw a batch of size {batch_size}. "
                             f"Current size: {self.size}")

        sample_indices = torch.randint(0, self.size, size=(batch_size,), device=self.device)
        
        return (
            self.states[sample_indices],
            self.actions[sample_indices].squeeze(-1),
            self.continuous_actions[sample_indices].squeeze(-1),
            self.next_states[sample_indices],
            self.energies[sample_indices].squeeze(-1),
            self.energies_before[sample_indices].squeeze(-1),
            self.dones[sample_indices].squeeze(-1),
            self.indices[sample_indices].squeeze(-1)
        )

    def get_size(self) -> int:
        """Returns the current number of transitions stored in the buffer."""
        return self.size

