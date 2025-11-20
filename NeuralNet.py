import torch
from torch import nn
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


class Actor(nn.Module):
    """
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
        std = torch.exp(log_std)

        return mu, std, discrete_scores

