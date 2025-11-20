"""
This module implements the Soft Actor-Critic (SAC) agent, which is responsible
for learning the optimal policy for constructing quantum circuits.
"""
import logging
import os
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
<<<<<<< Updated upstream
from torch.distributions import Categorical, Normal
=======
from NeuralNet import Critic, Actor
import os
from ReplayBuffer import ReplayBuffer
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import copy
import time
>>>>>>> Stashed changes

from NeuralNet import Actor, Critic
from QuantumStateEnv import QuantumStateEnv
from ReplayBuffer import ReplayBuffer
from reward import Rewards

<<<<<<< Updated upstream

class SAC_Agent:
    """
    The SAC agent, containing the actor and critic networks, and methods for
    training and action selection.
    """

    def __init__(self, cfg: Any, env: QuantumStateEnv, device: torch.device,
                 logger: logging.Logger, model_path: str, session: int, pred: bool = False):
        """
        Initializes the SAC agent.
        """
        self.cfg = cfg
        self.env = env
        self.device = device
        self.logger = logger
        self.model_path = model_path
        self.pred = pred
        self.session = session

        # Hyperparameters
        self.batch_size = cfg.SACparam["batch_size"]
        self.gamma = cfg.SACparam["gamma"]
        self.soft_update_factor = cfg.SACparam["soft_update_factor"]
        self.training_update_factor = cfg.SACparam["training_update_factor"]
        
        self.act_lower_bounds = -np.pi
        self.act_upper_bounds = np.pi

        # Initialize components
        self.replay_buffer = ReplayBuffer(cfg, device, env)
        self.rewards_calculator = Rewards(cfg, device)
        self._init_networks()
        self._init_optimizers()
        self._init_temperature_tuning()
=======
"""
Executing updates of the critic and actor networks and temperature tuning
according to the Soft Actor–Critic (SAC) algorithm, for a hybrid
discrete–continuous action space.
"""


class SAC_Agent:
    def __init__(self, cfg, env, device, logger, model_path, session, pred):
        """
        Initialize the SAC agent.

        Args:
            cfg: Configuration object/module with SAC, training, and reward parameters.
            env: Environment providing state_size and action_size attributes.
            device: Torch device ('cpu' or 'cuda').
            logger: Logger instance for diagnostics.
            model_path: Base directory for loading/storing models.
            session: Session identifier (used when loading models).
            pred: If True, load actor weights from disk instead of training from scratch.
        """
        # General configuration
        self.pred = pred
        self.model_path = model_path
        self.rwd = Rewards(cfg, device)
        self.device = device
        self.cfg = cfg
        self.environment = env

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.cfg, self.device, self.environment)

        # SAC hyperparameters
        self.batch_size = self.cfg.SACparam["batch_size"]
        self.gamma = self.cfg.SACparam["gamma"]
        self.lr_critic = self.cfg.SACparam["lr_critic"]
        self.lr_actor = self.cfg.SACparam["lr_actor"]
        self.lr_alpha_d = self.cfg.SACparam["lr_alpha_d"]
        self.lr_alpha_c = self.cfg.SACparam["lr_alpha_c"]
        self.soft_update_factor = self.cfg.SACparam["soft_update_factor"]
        self.training_update_factor = self.cfg.SACparam["training_update_factor"]
        self.target_entropy_deduction_d = self.cfg.SACparam["target_entropy_deduction_d"]
        self.target_entropy_deduction_c = self.cfg.SACparam["target_entropy_deduction_c"]
        self.target_entropy_end_value_d = self.cfg.SACparam["target_entropy_end_value_d"]
        self.target_entropy_end_value_c = self.cfg.SACparam["target_entropy_end_value_c"]
        self.max_gates = self.cfg.training["max_gates"]
        self.train_evaluate_ratio = self.cfg.training["train_evaluate_ratio"]

        # State and action dimensions
        self.state_dim = env.state_size
        self.action_dim = env.action_size

        # Decay factors for target entropy schedules
        self.decay_factor_c = self.cfg.SACparam["decay_factor_c"]
        self.decay_factor_d = self.cfg.SACparam["decay_factor_d"]
>>>>>>> Stashed changes

        # Training counter (used to decide when to update networks)
        self.training_count = 0

        # Continuous action (angle) bounds
        self.act_lower_bounds = -np.pi
        self.act_upper_bounds = np.pi

        # Precomputed index tensor for gathering Q-values of all discrete actions
        self.number_tensor = (
            torch.tensor([list(range(0, self.action_dim))])
            .repeat_interleave(self.batch_size, -2)
            .view(self.batch_size * self.action_dim, 1)
            .to(self.device)
        )

        # ---------------------------------------------------------------------
        # Networks and optimizers
        # ---------------------------------------------------------------------

        # Two critic networks for clipped double Q-learning.
        # Input dimension is state_dim + 1 (state plus continuous action).
        self.critic1 = Critic(
            self.cfg, input_dimension=self.state_dim + 1, output_dimension=self.action_dim
        ).to(self.device)
        self.critic2 = Critic(
            self.cfg, input_dimension=self.state_dim + 1, output_dimension=self.action_dim
        ).to(self.device)

        self.critic_optimiser1 = torch.optim.Adam(
            self.critic1.parameters(), lr=self.lr_critic
        )
        self.critic_optimiser2 = torch.optim.Adam(
            self.critic2.parameters(), lr=self.lr_critic
        )

        # Target critics for stable value estimation
        self.critic_target1 = Critic(
            self.cfg, input_dimension=self.state_dim + 1, output_dimension=self.action_dim
        ).to(self.device)
        self.critic_target2 = Critic(
            self.cfg, input_dimension=self.state_dim + 1, output_dimension=self.action_dim
        ).to(self.device)

        # Policy network:
        #   - outputs discrete action probabilities (Softmax),
        #   - means and stds for continuous actions.
        self.actor = Actor(
            self.cfg,
            input_dimension=self.state_dim,
            output_dimension=self.action_dim,
            output_activation=torch.nn.Softmax(dim=1),
        ).to(self.device)
        self.actor_optimiser = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr_actor
        )

        # Optionally load a pretrained actor for prediction mode
        if self.pred:
<<<<<<< Updated upstream
            self.load_models()

        self.training_count = 0
        self.rl_results = []

    def _init_networks(self):
        """Initializes the actor and critic networks."""
        state_dim = self.env.state_size
        action_dim = self.env.action_size

        # Critic networks
        critic_input_dim = state_dim + action_dim
        self.critic1 = Critic(self.cfg, input_dimension=critic_input_dim, output_dimension=action_dim).to(self.device)
        self.critic2 = Critic(self.cfg, input_dimension=critic_input_dim, output_dimension=action_dim).to(self.device)
        self.critic_target1 = Critic(self.cfg, input_dimension=critic_input_dim, output_dimension=action_dim).to(self.device)
        self.critic_target2 = Critic(self.cfg, input_dimension=critic_input_dim, output_dimension=action_dim).to(self.device)
        self.soft_update(self.critic_target1, self.critic1, tau=1.0) # Hard update initially
        self.soft_update(self.critic_target2, self.critic2, tau=1.0)

        # Actor network
        self.actor = Actor(self.cfg, input_dimension=state_dim, output_dimension=action_dim,
                           output_activation=torch.nn.Softmax(dim=1)).to(self.device)

    def _init_optimizers(self):
        """Initializes the optimizers for the networks."""
        self.critic_optimiser1 = torch.optim.Adam(self.critic1.parameters(), lr=self.cfg.SACparam["lr_critic"])
        self.critic_optimiser2 = torch.optim.Adam(self.critic2.parameters(), lr=self.cfg.SACparam["lr_critic"])
        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.SACparam["lr_actor"])

    def _init_temperature_tuning(self):
        """Initializes the temperature parameters (alpha) for entropy maximization."""
        # Discrete action temperature
        self.log_alpha_d = torch.zeros(1, requires_grad=True, device=self.device)
        self.log_alpha_optimiser_d = torch.optim.Adam([self.log_alpha_d], lr=self.cfg.SACparam["lr_alpha_d"])
        self.target_entropy_d = -self.cfg.SACparam["target_entropy_deduction_d"] * np.log(1 / self.env.action_size)

        # Continuous action temperature
        self.log_alpha_c = torch.zeros(1, requires_grad=True, device=self.device)
        self.log_alpha_optimiser_c = torch.optim.Adam([self.log_alpha_c], lr=self.cfg.SACparam["lr_alpha_c"])
        self.target_entropy_c = -self.cfg.SACparam["target_entropy_deduction_c"]

    def get_next_action(self, state: np.ndarray, evaluation_episode: bool = False, start_episodes: bool = False) -> Tuple[int, float]:
        """
        Selects an action based on the current policy and state.
        """
        if start_episodes:
            action = np.random.choice(range(self.env.action_size))
            angle_action = np.random.uniform(self.act_lower_bounds, self.act_upper_bounds)
            return action, angle_action

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, sigma, action_probs = self.actor(state_tensor)

        if evaluation_episode:
            # Deterministic action for evaluation
            action = torch.argmax(action_probs).item()
            angle_action = mu.squeeze(0)[action]
        else:
            # Stochastic action for training
            dist = Categorical(action_probs)
            action = dist.sample().item()
            normal_dist = Normal(mu.squeeze(0)[action], sigma.squeeze(0)[action])
            angle_action = normal_dist.sample()

        # Scale and clip the continuous action
        angle_action = torch.tanh(angle_action)
        angle_action = self.act_lower_bounds + 0.5 * (angle_action + 1.0) * (self.act_upper_bounds - self.act_lower_bounds)
        
        return action, angle_action.detach().cpu().item()

    def train_on_transition(self, state, action, action_c, next_state, energy, energy_before, done, e, index):
        """Adds a transition to the replay buffer and triggers training if ready."""
        self.replay_buffer.add_transition(state, action, action_c, next_state, energy, energy_before, done, index)
        
        self.training_count += 1
        if self.replay_buffer.size >= self.batch_size and self.training_count % self.training_update_factor == 0:
            self.train_networks(e)

    def train_networks(self, e: int):
        """Performs a single training step for all networks."""
        for _ in range(self.training_update_factor):
            states, actions, actions_c, next_states, energies, energies_before, dones, indices = self.replay_buffer.sample_batch(self.batch_size)
            rewards = self.rewards_calculator.get_reward(energies, energies_before, indices)

            # --- Critic Update ---
            critic_loss1, critic_loss2 = self._calculate_critic_loss(states, actions, actions_c, next_states, rewards, dones)
            self._update_network(self.critic_optimiser1, critic_loss1)
            self._update_network(self.critic_optimiser2, critic_loss2)

            # --- Actor and Alpha Update ---
            policy_loss, alpha_loss_d, alpha_loss_c, ent_d, ent_c = self._calculate_actor_and_alpha_loss(states)
            
            # Zero gradients for all three optimizers
            self.actor_optimiser.zero_grad()
            self.log_alpha_optimiser_d.zero_grad()
=======
            model_path_actor = os.path.join(
                self.model_path,
                f"training_session_{session}",
                f"actor_model_{session}.pt",
            )
            self.actor.load_state_dict(
                torch.load(model_path_actor, map_location=self.device)
            )

        # ---------------------------------------------------------------------
        # Temperature tuning (entropic regularization)
        # ---------------------------------------------------------------------

        # Discrete alpha (log parameterization to ensure non-negativity)
        self.log_alpha_d = torch.tensor([0.0]).to(self.device)
        self.log_alpha_d.requires_grad_(True)
        self.log_alpha_optimiser_d = torch.optim.SGD(
            [self.log_alpha_d], lr=self.lr_alpha_d
        )
        self.alpha_d = self.log_alpha_d.exp()

        # Continuous alpha (log parameterization)
        self.log_alpha_c = torch.tensor([0.0]).to(self.device)
        self.log_alpha_c.requires_grad_(True)
        self.log_alpha_optimiser_c = torch.optim.SGD(
            [self.log_alpha_c], lr=self.lr_alpha_c
        )
        self.alpha_c = self.log_alpha_c.exp()

        # Tracking of RL quantities per training episode
        self.rl_results = []

    # =====================================================================
    # Action selection
    # =====================================================================
    def get_next_action(
        self, state, evaluation_episode: bool = False, start_episodes: bool = False
    ):
        """
        Select the next action given the current state.

        Behavior:
            - Training (non-deterministic):
                Discrete action sampled from a categorical distribution;
                continuous action sampled from a Gaussian distribution.
            - Evaluation (deterministic):
                Discrete action = argmax over probabilities;
                continuous action = corresponding mean.
            - Initial episodes (start_episodes=True):
                Discrete and continuous actions sampled at random.

        Args:
            state: Current state as array-like (converted to a torch.Tensor).
            evaluation_episode: Use deterministic policy if True.
            start_episodes: Override with fully random actions if True.

        Returns:
            action: Discrete action index (int).
            angle_action: Continuous action (angle) as a numpy array.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu, sigma, action_probabilities = self.actor(state_tensor)

        # Non-deterministic action selection during training
        if not evaluation_episode and not start_episodes:
            action_probabilities_numpy = (
                action_probabilities.squeeze(0).detach().cpu().numpy()
            )
            # Ensure probabilities are valid
            action_probabilities_numpy = np.abs(action_probabilities_numpy)
            action_probabilities_numpy = (
                action_probabilities_numpy / np.sum(action_probabilities_numpy)
            )
            # Discrete action
            action = np.random.choice(
                range(self.action_dim), p=action_probabilities_numpy
            )

            # Continuous action (angle)
            pi_distribution = Normal(
                mu.squeeze(0)[action], sigma.squeeze(0)[action]
            )
            angle_action = pi_distribution.sample()

        # Deterministic action selection during evaluation
        elif evaluation_episode:
            action_probabilities_numpy = (
                action_probabilities.squeeze(0).detach().cpu().numpy()
            )
            action = int(np.argmax(action_probabilities_numpy))
            angle_action = mu.squeeze(0)[action]

        # Random actions during initial episodes
        else:
            action = np.random.choice(range(self.action_dim))
            choices = np.arange(-100, 101)
            num = np.random.choice(choices)
            angle_action = torch.tensor(
                np.pi * (num / 100), device=self.device
            )

        # Squash to (-1, 1) and rescale to [act_lower_bounds, act_upper_bounds]
        angle_action = torch.tanh(angle_action)
        angle_action = (
            self.act_lower_bounds
            + 0.5 * (angle_action + 1.0) * (self.act_upper_bounds - self.act_lower_bounds)
        )

        return action, angle_action.detach().cpu().numpy()

    # =====================================================================
    # Store transition and trigger training
    # =====================================================================
    def train_on_transition(
        self,
        state,
        action,
        action_c,
        next_state,
        energy,
        energy_before,
        done,
        e,
        index,
    ):
        """
        Store a transition in the replay buffer and perform network updates
        when enough transitions have been collected.

        Args:
            state: Current state representation.
            action: Discrete action index.
            action_c: Continuous action (angle).
            next_state: Next state representation.
            energy: Current energy.
            energy_before: Energy from the previous step.
            done: Boolean indicating whether the episode has terminated.
            e: Episode index (used for logging/tracking).
            index: Bond-distance index.
        """
        # Training counter increment (kept identical to original implementation)
        self.training_count = self.training_count + 1 + 1 / self.train_evaluate_ratio

        # Convert transition components to tensors
        state_tensor = torch.tensor(np.array(state)).float().to(self.device)
        action_tensor = torch.tensor(np.array(action)).to(self.device)
        action_c_tensor = torch.tensor(np.array(action_c)).to(self.device)
        next_state_tensor = torch.tensor(np.array(next_state)).float().to(self.device)
        energy_tensor = torch.tensor(np.array(energy)).float().to(self.device)
        energy_before_tensor = torch.tensor(np.array(energy_before)).float().to(
            self.device
        )
        done_tensor = torch.tensor(np.array(done)).to(self.device)
        index_tensor = torch.tensor(np.array(index)).to(self.device)

        # Add transition to replay buffer
        self.replay_buffer.add_transition(
            state_tensor,
            action_tensor,
            action_c_tensor,
            next_state_tensor,
            energy_tensor,
            energy_before_tensor,
            done_tensor,
            index_tensor,
        )

        # Perform updates only if enough transitions are available
        # and the update schedule condition is met
        if (
            self.replay_buffer.get_size() >= 1024
            and int(self.training_count) % self.training_update_factor == 0
        ):
            self.train_networks(e)

    # =====================================================================
    # Core update loop over sampled batches
    # =====================================================================
    def train_networks(self, e: int):
        """
        Perform several consecutive update steps, as specified by
        'training_update_factor'.
        """
        for _ in range(self.training_update_factor):
            (
                states_tensor,
                actions_tensor,
                actions_c_tensor,
                next_states_tensor,
                energy_tensor,
                energy_before_tensor,
                done_tensor,
                index_tensor,
            ) = self.replay_buffer.sample_batch(self.batch_size)

            # Compute rewards
            rewards_tensor = self.rwd.which_reward(
                energy_tensor,
                energy_before_tensor,
                done_tensor,
                index_tensor,
                tensor_form=True,
            )

            # ---------------- Critic update ----------------
            critic_loss1, critic_loss2 = self.critic_loss(
                states_tensor,
                actions_tensor,
                actions_c_tensor,
                next_states_tensor,
                rewards_tensor,
                done_tensor,
            )

            self.critic_optimiser1.zero_grad()
            self.critic_optimiser2.zero_grad()
            critic_loss1.backward()
            critic_loss2.backward()
            self.critic_optimiser1.step()
            self.critic_optimiser2.step()

            # Freeze critic parameters for policy update
            for p in self.critic1.parameters():
                p.requires_grad = False
            for p in self.critic2.parameters():
                p.requires_grad = False

            # ---------------- Policy update ----------------
            policy_loss = self.actor_loss(states_tensor)

            self.actor_optimiser.zero_grad()
            policy_loss.backward()
            self.actor_optimiser.step()

            # Unfreeze critic parameters
            for p in self.critic1.parameters():
                p.requires_grad = True
            for p in self.critic2.parameters():
                p.requires_grad = True

            # ---------------- Temperature tuning ----------------
            (
                alpha_loss_d,
                real_entropy_d,
                alpha_loss_c,
                real_entropy_c,
            ) = self.get_alpha_loss_and_entropy(states_tensor)

            # Discrete alpha update
            self.log_alpha_optimiser_d.zero_grad()
            alpha_loss_d.backward()
            self.log_alpha_optimiser_d.step()
            self.alpha_d = self.log_alpha_d.exp()

            # Continuous alpha update
>>>>>>> Stashed changes
            self.log_alpha_optimiser_c.zero_grad()

            # Perform a single backward pass for the combined loss
            (policy_loss + alpha_loss_d + alpha_loss_c).backward()

            # Step each optimizer
            self.actor_optimiser.step()
            self.log_alpha_optimiser_d.step()
            self.log_alpha_optimiser_c.step()
<<<<<<< Updated upstream
            
            # --- Target Network Update ---
            self.soft_update(self.critic_target1, self.critic1)
            self.soft_update(self.critic_target2, self.critic2)

            # --- Logging ---
            q_loss = 0.5 * (critic_loss1 + critic_loss2)
            alpha_d_val = self.log_alpha_d.exp().detach()
            alpha_c_val = self.log_alpha_c.exp().detach()
            self.rl_results.append((e, policy_loss.item(), q_loss.item(), alpha_d_val.item(), alpha_c_val.item(), ent_d.item(), ent_c.item()))

    def _calculate_critic_loss(self, states, actions, actions_c, next_states, rewards, dones) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the loss for the critic networks."""
        with torch.no_grad():
            next_mu, next_sigma, next_action_probs = self.actor(next_states)
            
            # Discrete part of the target: log probabilities for all actions
            # Add a small epsilon to prevent log(0)
            next_log_probs_d = torch.log(next_action_probs + 1e-8)

            # Continuous part of the target
            next_normal_dist = Normal(next_mu, next_sigma)
            next_angles = next_normal_dist.sample()
            next_log_probs_c = next_normal_dist.log_prob(next_angles).sum(axis=-1)
            
            # Target Q-values
            # We need to evaluate the critic for each next_state paired with its sampled next_angle
            # The input to the critic should be (batch_size, state_dim + 1)
            next_state_angle_cat = torch.cat([next_states, next_angles], dim=1)

            q_target1 = self.critic_target1(next_state_angle_cat)
            q_target2 = self.critic_target2(next_state_angle_cat)
            q_target = torch.min(q_target1, q_target2)
            
            alpha_d = self.log_alpha_d.exp().detach()
            alpha_c = self.log_alpha_c.exp().detach()

            # Reshape continuous log probs to (batch_size, 1) for broadcasting
            next_log_probs_c_expanded = next_log_probs_c.unsqueeze(-1)

            # All tensors inside this operation now have compatible shapes for broadcasting:
            # next_action_probs: (batch_size, action_dim)
            # q_target:          (batch_size, action_dim)
            # next_log_probs_d:  (batch_size, action_dim)
            # next_log_probs_c_expanded: (batch_size, 1) -> broadcasts to (batch_size, action_dim)
            expected_q_target = (next_action_probs * (q_target - alpha_d * next_log_probs_d - alpha_c * next_log_probs_c_expanded)).sum(dim=1)
            
            y = rewards + self.gamma * (1 - dones.int()) * expected_q_target

        # Current Q-values
        # The input to the critic should be (batch_size, state_dim + action_dim)
        state_action_cat = torch.cat([states, actions_c.unsqueeze(1).expand(-1, self.env.action_size)], dim=1)
        q1 = self.critic1(state_action_cat).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        q2 = self.critic2(state_action_cat).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # MSE Loss
        loss1 = F.mse_loss(q1, y)
        loss2 = F.mse_loss(q2, y)
        return loss1, loss2

    def _calculate_actor_and_alpha_loss(self, states) -> Tuple[torch.Tensor, ...]:
        """Calculates the loss for the actor network and the temperature alphas."""
        mu, sigma, action_probs = self.actor(states)
        
        # Freeze critic parameters
        for p in self.critic1.parameters(): p.requires_grad = False
        for p in self.critic2.parameters(): p.requires_grad = False

        # Continuous part
        normal_dist = Normal(mu, sigma)
        angles = normal_dist.rsample() # rsample for reparameterization trick
        log_probs_c = normal_dist.log_prob(angles).sum(axis=-1)

        # Discrete part - align with critic calculation
        dist = Categorical(action_probs)
        log_probs_d = torch.log(action_probs + 1e-8)

        # Q-values for the current policy
        # The input to the critic should be (batch_size, state_dim + action_dim)
        state_angle_cat = torch.cat([states, angles], dim=1)
        
        q1 = self.critic1(state_angle_cat)
        q2 = self.critic2(state_angle_cat)
        q = torch.min(q1, q2)
        
        # Policy loss
        alpha_d = self.log_alpha_d.exp().detach()
        alpha_c = self.log_alpha_c.exp().detach()
        
        # We need to sum over the action dimension for the policy loss
        # log_probs_d is (batch_size, action_dim), so we multiply by action_probs and sum
        # log_probs_c is (batch_size,), so we just use it
        policy_loss = (action_probs * (alpha_d * log_probs_d - q)).sum(dim=1) + alpha_c * log_probs_c
        policy_loss = policy_loss.mean()
=======
            self.alpha_c = self.log_alpha_c.exp()

            # ---------------- Soft update of target networks ----------------
            self.soft_update(self.critic_target1, self.critic1)
            self.soft_update(self.critic_target2, self.critic2)

            # Q-loss tracking (kept identical to original code)
            Q_loss = 0.5 * (critic_loss1 + critic_loss1)

        # Store RL quantities for this episode
        self.rl_results.append(
            (
                e,
                policy_loss.detach().cpu().numpy(),
                Q_loss.detach().cpu().numpy(),
                self.alpha_d.detach().cpu().numpy(),
                self.alpha_c.detach().cpu().numpy(),
                real_entropy_d.detach().cpu().numpy(),
                real_entropy_c.detach().cpu().numpy(),
            )
        )

    # =====================================================================
    # Critic losses
    # =====================================================================
    def critic_loss(
        self,
        states_tensor,
        actions_tensor,
        actions_c_tensor,
        next_states_tensor,
        rewards_tensor,
        done_tensor,
    ):
        """
        Compute critic losses for the two Q-networks.
        """
        with torch.no_grad():
            (
                action_probabilities,
                log_action_probabilities,
                angle_action,
                log_angle_action,
            ) = self.get_pi_and_logpi(next_states_tensor)

            # Q-values for next states and all discrete actions (target critics)
            q_values_target1 = (
                self.critic_target1(
                    torch.cat(
                        (
                            next_states_tensor.repeat_interleave(
                                self.action_dim, -2
                            ),
                            angle_action.view(
                                self.batch_size * self.action_dim, 1
                            ),
                        ),
                        dim=1,
                    )
                )
                .gather(1, self.number_tensor)
                .view(self.batch_size, self.action_dim)
            )

            q_values_target2 = (
                self.critic_target2(
                    torch.cat(
                        (
                            next_states_tensor.repeat_interleave(
                                self.action_dim, -2
                            ),
                            angle_action.view(
                                self.batch_size * self.action_dim, 1
                            ),
                        ),
                        dim=1,
                    )
                )
                .gather(1, self.number_tensor)
                .view(self.batch_size, self.action_dim)
            )

            # Clipped double Q: use the element-wise minimum
            q_values_targets_min = torch.min(q_values_target1, q_values_target2)

            # Soft state value function
            soft_state_value_function = (
                action_probabilities
                * (
                    q_values_targets_min
                    - self.alpha_d * log_action_probabilities
                    - self.alpha_c * log_angle_action
                )
            ).sum(dim=1)

            # Bellman backup operator
            bellman_backup_operator = (
                rewards_tensor + ~done_tensor * self.gamma * soft_state_value_function
            )

        # Q-values at current states for selected discrete actions
        Q_values1 = (
            self.critic1(
                torch.cat((states_tensor, actions_c_tensor.unsqueeze(1)), dim=1)
            )
            .gather(1, actions_tensor.unsqueeze(-1))
            .squeeze(-1)
        )
        Q_values2 = (
            self.critic2(
                torch.cat((states_tensor, actions_c_tensor.unsqueeze(1)), dim=1)
            )
            .gather(1, actions_tensor.unsqueeze(-1))
            .squeeze(-1)
        )

        # Mean squared errors (0.5 factor kept from original implementation)
        critic_MSE_1 = 0.5 * (Q_values1 - bellman_backup_operator) ** 2
        critic_MSE_2 = 0.5 * (Q_values2 - bellman_backup_operator) ** 2

        critic_loss1 = critic_MSE_1.mean()
        critic_loss2 = critic_MSE_2.mean()

        return critic_loss1, critic_loss2

    # =====================================================================
    # Actor loss
    # =====================================================================
    def actor_loss(self, states_tensor):
        """
        Compute the policy loss for the actor network.
        """
        (
            action_probabilities,
            log_action_probabilities,
            angle_action,
            log_angle_action,
        ) = self.get_pi_and_logpi(states_tensor)

        # Q-values for all discrete actions and sampled continuous actions
        q_values1 = (
            self.critic1(
                torch.cat(
                    (
                        states_tensor.repeat_interleave(self.action_dim, -2),
                        angle_action.view(
                            self.batch_size * self.action_dim, 1
                        ),
                    ),
                    dim=1,
                )
            )
            .gather(1, self.number_tensor)
            .view(self.batch_size, self.action_dim)
        )

        q_values2 = (
            self.critic2(
                torch.cat(
                    (
                        states_tensor.repeat_interleave(self.action_dim, -2),
                        angle_action.view(
                            self.batch_size * self.action_dim, 1
                        ),
                    ),
                    dim=1,
                )
            )
            .gather(1, self.number_tensor)
            .view(self.batch_size, self.action_dim)
        )

        # Policy loss (per original implementation)
        policy_loss = (1 / self.batch_size) * (
            (
                action_probabilities
                * (
                    self.alpha_d * log_action_probabilities
                    + self.alpha_c * log_angle_action
                    - torch.min(q_values1, q_values2)
                )
            ).sum(dim=1)
        ).sum()

        return policy_loss

    # =====================================================================
    # Temperature losses and entropies
    # =====================================================================
    def get_alpha_loss_and_entropy(self, states_tensor):
        """
        Compute the losses for temperature tuning and the current entropies
        of the discrete and continuous components.
        """
        (
            action_probabilities,
            log_action_probabilities,
            angle_action,
            log_angle_action,
        ) = self.get_pi_and_logpi(states_tensor)

        # Discrete part
        alpha_loss_d = -(
            1 / self.batch_size
        ) * (
            self.log_alpha_d.exp()
            * action_probabilities.detach()
            * (log_action_probabilities + self.target_entropy_d).detach()
        ).sum()
        real_entropy_d = -(
            1 / self.batch_size
        ) * (action_probabilities * log_action_probabilities).sum()

        # Continuous part
        alpha_loss_c = -(
            1 / self.batch_size
        ) * (
            self.log_alpha_c.exp()
            * action_probabilities.detach()
            * (log_angle_action + self.target_entropy_c).detach()
        ).sum()
        real_entropy_c = -(
            1 / self.batch_size
        ) * (action_probabilities * log_angle_action).sum()

        return alpha_loss_d, real_entropy_d, alpha_loss_c, real_entropy_c

    # =====================================================================
    # Policy distribution and log-probabilities
    # =====================================================================
    def get_pi_and_logpi(self, states_tensor):
        """
        Compute action probabilities, log-probabilities, sampled continuous
        actions, and corresponding log-probabilities, in the form
        required for SAC updates.

        From the actor, obtain:
            - vec(p)       : action_probabilities
            - vec(mu)      : means of Gaussians
            - vec(sigma)   : standard deviations of Gaussians

        This method returns:
            - vec(p),
            - vec(log p),
            - vec(sampled continuous action),
            - vec(log-probabilities of sampled continuous actions)
        where vectors contain one entry per discrete action.
        """
        # Discrete distribution
        mu, std, action_probabilities = self.actor.forward(states_tensor)
        z = 1e-10
        log_action_probabilities = torch.log(action_probabilities + z)

        # Continuous distributions
        pi_distributions = Normal(mu, std)
        angle_action = pi_distributions.rsample()
        logp_pi = pi_distributions.log_prob(angle_action)

        # Tanh-squash correction for log-probabilities
        logp_pi -= (
            2 * (np.log(2) - angle_action - F.softplus(-2 * angle_action))
        )

        # Apply tanh squashing and rescale to angle bounds
        angle_action = torch.tanh(angle_action)
        angle_action = (
            self.act_lower_bounds
            + 0.5 * (angle_action + 1.0) * (self.act_upper_bounds - self.act_lower_bounds)
        )

        return action_probabilities, log_action_probabilities, angle_action, logp_pi

    # =====================================================================
    # Target network soft update
    # =====================================================================
    def soft_update(self, target_model, origin_model):
        """
        Soft update of target network parameters.

        target_param ← τ * local_param + (1 − τ) * target_param
        """
        for target_param, local_param in zip(
            target_model.parameters(), origin_model.parameters()
        ):
            target_param.data.copy_(
                self.soft_update_factor * local_param.data
                + (1 - self.soft_update_factor) * target_param.data
            )

    # =====================================================================
    # Accessors and utility methods
    # =====================================================================
    def rl_quantities(self):
        """
        Return the tracked RL results.

        Returns:
            List of tuples containing RL statistics per episode.
        """
        return self.rl_results
>>>>>>> Stashed changes

    def dynamic_values_for_reward_update(self, E, index):
        """
        Update dynamic quantities used inside the reward function.
        """
        self.rwd.dynamic_values_for_reward_update(E, index)

<<<<<<< Updated upstream
        # Unfreeze critic parameters
        for p in self.critic1.parameters(): p.requires_grad = True
        for p in self.critic2.parameters(): p.requires_grad = True

        # Alpha losses
        # We need the expectation of log_probs, so we multiply
        alpha_loss_d = -(log_probs_d + self.target_entropy_d).mean()
        alpha_loss_c = -(log_probs_c + self.target_entropy_c).mean()
        entropy_d = -log_probs_d.mean()
        entropy_c = -log_probs_c.mean()

        return policy_loss, alpha_loss_d, alpha_loss_c, entropy_d, entropy_c

    def _update_network(self, optimizer, loss):
        """Generic function to perform a gradient update step."""
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
        optimizer.step()

    def soft_update(self, target_net, source_net, tau=None):
        """Performs a soft update of the target network's parameters."""
        if tau is None:
            tau = self.soft_update_factor
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def save_models(self):
        """Saves the actor and critic models."""
        self.logger.info("    ... saving models ...")
        torch.save(self.actor.state_dict(), os.path.join(self.model_path, f"actor_{self.session}.pth"))
        torch.save(self.critic1.state_dict(), os.path.join(self.model_path, f"critic1_{self.session}.pth"))
        torch.save(self.critic2.state_dict(), os.path.join(self.model_path, f"critic2_{self.session}.pth"))

    def load_models(self):
        """Loads pre-trained models."""
        self.logger.info("    ... loading models ...")
        self.actor.load_state_dict(torch.load(os.path.join(self.model_path, f"actor_{self.session}.pth")))
        self.critic1.load_state_dict(torch.load(os.path.join(self.model_path, f"critic1_{self.session}.pth")))
        self.critic2.load_state_dict(torch.load(os.path.join(self.model_path, f"critic2_{self.session}.pth")))
=======
    # ---------------------------------------------------------------------
    # Target entropy schedule
    # ---------------------------------------------------------------------
    def update_target_entropy(self, step, decay):
        """
        Update the target entropies for discrete and continuous components
        according to exponential schedules.

        Args:
            step: Global step index.
            decay: Total decay horizon (e.g., max_episodes * max_gates).
        """
        h_start_d = np.log(self.environment.action_size) - self.target_entropy_deduction_d
        h_end_d = self.target_entropy_end_value_d

        h_start_c = np.log(2) - self.target_entropy_deduction_c
        h_end_c = self.target_entropy_end_value_c

        self.target_entropy_d = h_end_d + (h_start_d - h_end_d) * np.exp(
            -self.decay_factor_d * (step / decay)
        )
        self.target_entropy_c = h_end_c + (h_start_c - h_end_c) * np.exp(
            -self.decay_factor_c * (step / decay)
        )

    # ---------------------------------------------------------------------
    # Model saving
    # ---------------------------------------------------------------------
    def save_policy(self, x, session_folder):
        """
        Save the actor network parameters to disk.

        Args:
            x: Training session index.
            session_folder: Path to the folder where the model is stored.
        """
        actor_model_path = os.path.join(session_folder, f"actor_model_{x}.pt")
        torch.save(self.actor.state_dict(), actor_model_path)
>>>>>>> Stashed changes
