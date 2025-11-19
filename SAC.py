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
from torch.distributions import Categorical, Normal

from NeuralNet import Actor, Critic
from QuantumStateEnv import QuantumStateEnv
from ReplayBuffer import ReplayBuffer
from reward import Rewards


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

        if self.pred:
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
            self.log_alpha_optimiser_c.zero_grad()

            # Perform a single backward pass for the combined loss
            (policy_loss + alpha_loss_d + alpha_loss_c).backward()

            # Step each optimizer
            self.actor_optimiser.step()
            self.log_alpha_optimiser_d.step()
            self.log_alpha_optimiser_c.step()
            
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