import numpy as np
import torch
from typing import Tuple


"""
Replay buffer: initialization, storing transitions, and sampling batches.
"""


class ReplayBuffer:
    """
    Simple replay buffer storing transitions on a given device.

    It stores:
        state, action (discrete), action_c (continuous),
        next_state, energy, energy_before, done, index.
    """

    def __init__(self, cfg, device: torch.device, env) -> None:
        """
        Initialize an empty replay buffer with fixed capacity.

        Args:
            cfg: Config object/module with RBparam["capacity"].
            device: Torch device to store all tensors on.
            env: Environment providing `state_size`.
        """
        self.device = device
        self.env = env
        self.cfg = cfg

        self.capacity: int = self.cfg.RBparam["capacity"]

        # Preallocate all tensors on the correct device
        self.state = torch.zeros(
            (self.capacity, self.env.state_size),
            dtype=torch.float32,
            device=self.device,
        )
        self.action = torch.zeros(
            (self.capacity, 1),
            dtype=torch.int64,
            device=self.device,
        )
        self.action_c = torch.zeros(
            (self.capacity, 1),
            dtype=torch.float32,
            device=self.device,
        )
        self.next_state = torch.zeros(
            (self.capacity, self.env.state_size),
            dtype=torch.float32,
            device=self.device,
        )
        self.energy = torch.zeros(
            (self.capacity, 1),
            dtype=torch.float32,
            device=self.device,
        )
        self.energy_before = torch.zeros(
            (self.capacity, 1),
            dtype=torch.float32,
            device=self.device,
        )
        self.done = torch.zeros(
            (self.capacity, 1),
            dtype=torch.bool,
            device=self.device,
        )
        self.index = torch.zeros(
            (self.capacity, 1),
            dtype=torch.int64,
            device=self.device,
        )

        # Pointer to the next write position, and current size
        self.next_index: int = 0
        self.count: int = 0

        # Last sampled indices (if needed elsewhere)
        self.indices = None

    def add_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        action_c: torch.Tensor,
        next_state: torch.Tensor,
        energy: torch.Tensor,
        energy_before: torch.Tensor,
        done: torch.Tensor,
        index: torch.Tensor,
    ) -> None:
        """
        Add a single transition to the replay buffer.

        If the buffer is full, the oldest transition is overwritten.
        """
        # Store transition at current write index
        self.state[self.next_index] = state
        self.action[self.next_index] = action
        self.action_c[self.next_index] = action_c
        self.next_state[self.next_index] = next_state
        self.energy[self.next_index] = energy
        self.energy_before[self.next_index] = energy_before
        self.done[self.next_index] = done
        self.index[self.next_index] = index

        # Move write index forward (ring buffer)
        self.next_index = (self.next_index + 1) % self.capacity

        # Increase count up to capacity
        if self.count < self.capacity:
            self.count += 1

    def sample_batch(
        self,
        batch_size: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Sample a random batch of transitions.

        Returns:
            state, action, action_c, next_state,
            energy, energy_before, done, index
        """
        # Sample indices for the batch uniformly
        self.indices = torch.randint(
            0, self.count, size=(batch_size,), device=self.device
        )

        state = self.state[self.indices]
        action = self.action[self.indices].squeeze(-1)
        action_c = self.action_c[self.indices].squeeze(-1)
        next_state = self.next_state[self.indices]
        energy = self.energy[self.indices].squeeze(-1)
        energy_before = self.energy_before[self.indices].squeeze(-1)
        done = self.done[self.indices].squeeze(-1)
        index = self.index[self.indices].squeeze(-1)

        # Return order is identical to the original implementation
        return (
            state,
            action,
            action_c,
            next_state,
            energy,
            energy_before,
            done,
            index,
        )

    def get_size(self) -> int:
        """
        Returns:
            Current number of stored transitions (<= capacity).
        """
        return self.count
