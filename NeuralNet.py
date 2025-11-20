"""
This module defines the neural network architectures for the Actor and Critic models
used in the Soft Actor-Critic (SAC) algorithm.
"""
import torch
from torch import nn
<<<<<<< Updated upstream

class Critic(nn.Module):
    """
    A Q-network (Critic) that estimates the Q-value for a given state and continuous action.

    The network outputs a vector of Q-values, one for each discrete action.
    q(s, c) -> (q(s, a_1, c), ..., q(s, a_|A|, c))
    """

    def __init__(self, cfg, input_dimension: int, output_dimension: int, output_activation: nn.Module = nn.Identity()):
        """
        Initializes the Critic network.

        Args:
            cfg: Configuration object with neural network parameters.
            input_dimension: The dimension of the input state.
            output_dimension: The number of discrete actions.
            output_activation: The activation function for the output layer.
        """
        super().__init__()
        hidden_layers = cfg.NeuralNet["hidden_layers_critic"]
        
        layers = []
        layer_sizes = [input_dimension] + hidden_layers
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.LayerNorm(layer_sizes[i+1]))
            layers.append(nn.ReLU())
        
        # Add a final layer for the output
        layers.append(nn.Linear(layer_sizes[-1], output_dimension))
        layers.append(output_activation)
            
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the network.

        Args:
            x: The input tensor representing the state.

        Returns:
            A tensor of Q-values for each discrete action.
        """
        return self.network(x)
=======
from typing import Sequence, Tuple


class Critic(nn.Module):
    """
    Q-network: q(s, continuous action = c) -> q-values for each discrete action.

    Input:
        x: state (optionally concatenated with the continuous action),
           shape (batch_size, input_dimension)

    Output:
        qvalues: Q-values for all discrete actions,
                 shape (batch_size, output_dimension)
    """

    def __init__(
        self,
        cfg,
        input_dimension: int,
        output_dimension: int,
        output_activation=nn.Identity(),
    ) -> None:
        super().__init__()

        self.hidden_layers: Sequence[int] = cfg.NeuralNet["hidden_layers_critic"]
        if len(self.hidden_layers) == 0:
            raise ValueError("cfg.NeuralNet['hidden_layers_critic'] must not be empty.")

        # Stack of hidden layers
        self.layers = nn.ModuleList()
        in_features = input_dimension
        for hidden_size in self.hidden_layers:
            self.layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size

        # Output layer
        self.output_layer = nn.Linear(self.hidden_layers[-1], output_dimension)
        self.output_activation = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        qvalues = self.output_activation(self.output_layer(x))
        return qvalues
>>>>>>> Stashed changes


class Actor(nn.Module):
    """
<<<<<<< Updated upstream
    A policy-network (Actor) that outputs the policy for a given state.

    The policy consists of:
    - A probability distribution over discrete actions.
    - The mean (mu) and standard deviation (sigma) for the continuous action distribution
      corresponding to each discrete action.
    pi(s) -> (p, mu, sigma)
    """

    def __init__(self, cfg, input_dimension: int, output_dimension: int, output_activation: nn.Module = nn.Identity()):
        """
        Initializes the Actor network.

        Args:
            cfg: Configuration object with neural network parameters.
            input_dimension: The dimension of the input state.
            output_dimension: The number of discrete actions.
            output_activation: The activation function for the output layer.
        """
        super().__init__()
        hidden_layers = cfg.NeuralNet["hidden_layers_actor"]
        
        layers = []
        layer_sizes = [input_dimension] + hidden_layers
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.LayerNorm(layer_sizes[i+1]))
            layers.append(nn.ReLU())
            
        self.backbone = nn.Sequential(*layers)
        
        last_hidden_size = hidden_layers[-1]
        self.discrete_probs_layer = nn.Linear(last_hidden_size, output_dimension)
        self.mu_layer = nn.Linear(last_hidden_size, output_dimension)
        self.sigma_layer = nn.Linear(last_hidden_size, output_dimension)
        self.output_activation = output_activation

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass through the network.

        Args:
            x: The input tensor representing the state.

        Returns:
            A tuple containing:
            - mu: The mean of the continuous action distribution.
            - std: The standard deviation of the continuous action distribution.
            - discrete_probs: The probability distribution over discrete actions.
        """
        x = self.backbone(x)

        # Output p, mu, log_sigma
        discrete_probs = self.output_activation(self.discrete_probs_layer(x))
        mu = torch.tanh(self.mu_layer(x)) # Use tanh to bound mu between -1 and 1
        log_std = self.sigma_layer(x)
        
        # Clamp log_std for stability and exponentiate to get std
        log_std = torch.clamp(log_std, min=-5, max=2) # Adjusted clamp range
=======
    Policy network:

        pi(s) -> (mu, std, discrete_scores)

    For each discrete action a_i:
        - mu_i, std_i: parameters of the continuous distribution
                       associated with this action (e.g., angle)
        - discrete_scores_i: unnormalized score / logit for the discrete
                             action (Softmax is applied externally).
    """

    def __init__(
        self,
        cfg,
        input_dimension: int,
        output_dimension: int,
        output_activation=nn.Identity(),
    ) -> None:
        super().__init__()

        self.hidden_layers: Sequence[int] = cfg.NeuralNet["hidden_layers_actor"]
        if len(self.hidden_layers) == 0:
            raise ValueError("cfg.NeuralNet['hidden_layers_actor'] must not be empty.")

        # Stack of hidden layers
        self.layers = nn.ModuleList()
        in_features = input_dimension
        for hidden_size in self.hidden_layers:
            self.layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size

        # Three heads: discrete scores, mu, log_sigma
        last_hidden_size = self.hidden_layers[-1]
        self.discrete_probs_layer = nn.Linear(last_hidden_size, output_dimension)
        self.mu_layer = nn.Linear(last_hidden_size, output_dimension)
        self.sigma_layer = nn.Linear(last_hidden_size, output_dimension)

        self.output_activation = output_activation

        # Clamping range for log_std
        self._log_std_min = -20.0
        self._log_std_max = 2.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x = torch.relu(layer(x))

        # Discrete part (scores / logits)
        discrete_scores = self.output_activation(self.discrete_probs_layer(x))

        # Continuous part (Gaussian parameters)
        mu = self.mu_layer(x)
        log_std = self.sigma_layer(x)
        log_std = torch.clamp(log_std, min=self._log_std_min, max=self._log_std_max)
>>>>>>> Stashed changes
        std = torch.exp(log_std)

        return mu, std, discrete_scores

