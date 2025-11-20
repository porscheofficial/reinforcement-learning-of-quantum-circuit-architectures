"""
This module defines the reward functions used to guide the reinforcement learning
agent. The rewards are designed to encourage the agent to find quantum circuits
that produce low energies.
"""
from typing import Any, Dict

import numpy as np
import torch
from typing import Dict, List


<<<<<<< Updated upstream

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


=======
"""
Reward functions.
"""


class Rewards:
    """
    Implements the reward scheme for the RL agent.

    Currently, only the 'exp_moving' reward is used. It depends on:
        - current energy E
        - previous energy E_before
        - the bond-distance index
        - running estimates (E_1, E_2) of good energies per bond distance
    """

    def __init__(self, cfg, device: torch.device) -> None:
        """
        Args:
            cfg: Config object/module with sections 'reward' and 'characteristics'.
            device: Torch device for tensor computations in the reward.
        """
        self.cfg = cfg
        self.device = device

        # Hyperparameters for the reward
        self.sigma_min: float = self.cfg.reward["sigma_min"]
        self.c_lin: float = self.cfg.reward["c_lin"]
        self.c_exp: float = self.cfg.reward["c_exp"]

        # Per-bond-distance energy histories (two moving sets)
        self.E_1: Dict[int, List[float]] = {}
        self.E_2: Dict[int, List[float]] = {}

        # Determine number of bond-distance points
        bond_grid = np.arange(
            self.cfg.characteristics["start_bond_distance"],
            self.cfg.characteristics["end_bond_distance"],
            self.cfg.characteristics["step_size_bond_distance"],
        )
        num_bond_points = len(bond_grid)

        # Initialize with initial_energy for all bond-distance indices
        initial_energy = self.cfg.characteristics["initial_energy"]
        for key in range(num_bond_points):
            self.E_1[key] = [initial_energy]
            self.E_2[key] = [initial_energy]

    def which_reward(
        self,
        E,
        E_before,
        done: bool,
        index,
        tensor_form: bool,
    ):
        """
        Select the reward function according to cfg.reward["reward"].

        Args:
            E: Current energy (float or Tensor).
            E_before: Previous energy (float or Tensor).
            done: Whether the episode is finished (not used in current scheme).
            index: Bond-distance index (int or Tensor of ints).
            tensor_form: If True, return a torch.Tensor, otherwise a float/np.float.

        Returns:
            Reward in the same “type world” as E/E_before (Tensor or float-like).
        """
        if self.cfg.reward["reward"] == "exp_moving":
            reward = self.exp_moving(E, E_before, done, index, tensor_form)
        else:
            # Fallback behavior: default to exp_moving if another key is set.
            reward = self.exp_moving(E, E_before, done, index, tensor_form)

        return reward

    def dynamic_values_for_reward_update(self, E: float, index: int) -> None:
        """
        Update the moving sets E_1 and E_2 for the given bond-distance index.

        E_1 and E_2 store the best (lowest) energies seen so far, with lengths
        capped by mu_average and sigma_average, respectively.
        """
        # Best-known energies for this bond distance
        E1_list = self.E_1[index]
        E2_list = self.E_2[index]

        # Update E_1 if E is better (smaller) than the worst stored in E_1
        if E < E1_list[np.argmax(E1_list)]:
            if len(E1_list) == self.cfg.reward["mu_average"]:
                # Replace current worst by new energy
                E1_list[np.argmax(E1_list)] = E
            else:
                # Still room to grow: append
                E1_list.append(E)

        # Otherwise, try to update E_2 (second-level best energies)
        elif E < E2_list[np.argmax(E2_list)]:
            if len(E2_list) == self.cfg.reward["sigma_average"]:
                E2_list[np.argmax(E2_list)] = E
            else:
                E2_list.append(E)

        # Lists are mutated in-place; no return value needed.

    def exp_moving(
        self,
        E,
        E_before,
        done: bool,   # kept for signature compatibility, not used
        index,
        tensor_form: bool,
    ):
        """
        Exponential moving reward.

        For tensor_form=True:
            E, E_before: Tensors of shape (batch_size,)
            index: Tensor of shape (batch_size,) with integer indices

        For tensor_form=False:
            E, E_before: scalar floats
            index: scalar int
        """
        if tensor_form:
            # Compute mu and sigma per index on CPU via numpy, then move to device
            mu_vals = [np.mean(self.E_1[i.item()]) for i in index]
            mu = torch.tensor(mu_vals, device=self.device)

            sigma_vals = [
                np.abs(np.mean(self.E_1[i.item()]) - np.mean(self.E_2[i.item()]))
                for i in index
            ]
            sigma = torch.tensor(sigma_vals, device=self.device) + self.sigma_min

            ex1 = torch.exp(-(E - mu) / sigma)
            ex2 = torch.exp(-(E_before - mu) / sigma)
            lin = (E - E_before)
        else:
            # Single bond-distance index, scalar energies
            mu = np.mean(self.E_1[index])
            sigma = np.abs(mu - np.mean(self.E_2[index])) + self.sigma_min

            ex1 = np.exp(-(E - mu) / sigma)
            ex2 = np.exp(-(E_before - mu) / sigma)
            lin = (E - E_before)

        # Same formula as in the original code
        return self.c_exp * (ex1 - ex2) - self.c_lin * lin
>>>>>>> Stashed changes



