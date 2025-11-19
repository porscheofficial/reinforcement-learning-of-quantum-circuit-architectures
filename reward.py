"""
This module defines the reward functions used to guide the reinforcement learning
agent. The rewards are designed to encourage the agent to find quantum circuits
that produce low energies.
"""
from typing import Any, Dict

import numpy as np
import torch


class Rewards:
    """
    A class to calculate rewards based on the energy improvement achieved by the agent.
    """

    def __init__(self, cfg: Any, device: torch.device):
        """
        Initializes the Rewards class.

        Args:
            cfg: A configuration object containing reward parameters.
            device: The torch device (CPU or CUDA).
        """
        self.cfg = cfg
        self.device = device
        self.sigma_min = cfg.reward["sigma_min"]
        self.c_lin = cfg.reward["c_lin"]
        self.c_exp = cfg.reward["c_exp"]
        
        # Dictionaries to store energy history for normalization
        self.E_1: Dict[int, float] = {}
        self.E_2: Dict[int, float] = {}

        num_bond_steps = len(np.arange(
            cfg.characteristics["start_bond_distance"],
            cfg.characteristics["end_bond_distance"],
            cfg.characteristics["step_size_bond_distance"]
        ))
        
        for key in range(num_bond_steps):
            self.E_1[key] = cfg.characteristics["initial_energy"]
            self.E_2[key] = cfg.characteristics["initial_energy"]

    def get_reward(self, energy: torch.Tensor, energy_before: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reward for a batch of transitions.

        The reward is a combination of a linear and an exponential term, designed
        to heavily penalize energy increases and reward energy decreases.

        Args:
            energy: A tensor of the current energies.
            energy_before: A tensor of the energies before the last action.
            index: A tensor of indices corresponding to the bond distance.

        Returns:
            A tensor of calculated rewards.
        """
        delta_E = energy - energy_before
        
        # Normalize the energy difference
        sigma = self._get_sigma(index)
        normalized_delta_E = delta_E / sigma

        # Calculate linear and exponential reward components
        linear_reward = -self.c_lin * normalized_delta_E
        exp_reward = -self.c_exp * torch.exp(normalized_delta_E)

        reward = linear_reward + exp_reward
        
        # Update the energy history for future normalization
        self._update_energy_history(energy, index)
        
        return reward

    def _get_sigma(self, index: torch.Tensor) -> torch.Tensor:
        """
        Calculates the standard deviation of energy improvements for normalization.
        """
        E_1_tensor = torch.tensor([self.E_1[int(i)] for i in index], device=self.device)
        E_2_tensor = torch.tensor([self.E_2[int(i)] for i in index], device=self.device)
        
        sigma_squared = E_2_tensor - E_1_tensor**2
        # Ensure sigma is not too small to avoid division by zero
        sigma = torch.sqrt(torch.max(sigma_squared, torch.tensor(self.sigma_min**2, device=self.device)))
        
        return sigma

    def _update_energy_history(self, energy: torch.Tensor, index: torch.Tensor):
        """
        Updates the running mean and mean square of the energy for normalization.
        """
        alpha = self.cfg.reward["alpha"]
        
        for i, e in zip(index, energy):
            key = int(i)
            self.E_1[key] = (1 - alpha) * self.E_1[key] + alpha * e
            self.E_2[key] = (1 - alpha) * self.E_2[key] + alpha * e**2





