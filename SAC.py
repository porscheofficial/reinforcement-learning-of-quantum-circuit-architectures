import numpy as np
import torch
import torch.nn.functional as F
from NeuralNet import Critic, Actor
import os
from ReplayBuffer import ReplayBuffer
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import copy
import time

from reward import Rewards

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
            self.log_alpha_optimiser_c.zero_grad()
            alpha_loss_c.backward()
            self.log_alpha_optimiser_c.step()
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

    def dynamic_values_for_reward_update(self, E, index):
        """
        Update dynamic quantities used inside the reward function.
        """
        self.rwd.dynamic_values_for_reward_update(E, index)

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
