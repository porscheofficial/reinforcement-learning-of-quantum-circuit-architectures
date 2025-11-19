"""
Configuration for Training Run 1: LiH4 Molecule, 2.2-2.3 Å bond distance.
"""

# --- Training Parameters ---
training = {
    "max_episodes": 30000,          # Total number of episodes for training
    "max_gates": 12,                # Maximum number of gates in the quantum circuit
    "train_evaluate_ratio": 10,     # Frequency of evaluation episodes (1 in every 10)
    "random_episodes": 100,         # Number of initial episodes with random actions
    "runs": 4,                      # Number of independent training sessions to run
    "results_folder_name": "LiH4_22" # Name for the folder to save results
}

# --- System and Environment Characteristics ---
characteristics = {
    "system": "LiH4",               # The quantum system to simulate (Lithium Hydride)
    "hf_start": "HF",               # Initial state of the quantum system (Hartree-Fock)
    "start_bond_distance": 2.2,     # Starting bond distance for the molecule in Angstrom
    "end_bond_distance": 2.3,       # Ending bond distance for the molecule in Angstrom
    "step_size_bond_distance": 0.2, # Step size for bond distance exploration
    "initial_energy": -7,           # An initial reference energy value
}

# --- Reward Function Parameters ---
reward = {
    "sigma_min": 0.01,              # Minimum value for the energy standard deviation (for stability)
    "c_lin": 0.1,                   # Coefficient for the linear part of the reward
    "c_exp": 5,                     # Coefficient for the exponential part of the reward
    "alpha": 0.1                    # Smoothing factor for updating energy statistics
}

# --- Gaussian Encoding for Bond Distance ---
gaussian_encoding = {
    "number_of_embeddings": 6,      # Number of Gaussian functions for encoding bond distance
    "start_interval": 1,            # Start of the interval for Gaussian means
    "end_interval": 4               # End of the interval for Gaussian means
}

# --- Soft Actor-Critic (SAC) Hyperparameters ---
SACparam = {
    "batch_size": 512,              # Number of samples in each training batch
    "gamma": 1,                     # Discount factor for future rewards
    "lr_critic": 0.001,             # Learning rate for the critic networks
    "lr_actor": 0.001,              # Learning rate for the actor network
    "lr_alpha_d": 0.003,            # Learning rate for the discrete temperature (alpha)
    "lr_alpha_c": 0.003,            # Learning rate for the continuous temperature (alpha)
    "soft_update_factor": 0.005,    # Interpolation factor for target network soft updates
    "training_update_factor": 50,   # Frequency of network updates
    "target_entropy_deduction_d": 0.1, # Target entropy for discrete actions
    "target_entropy_deduction_c": 0.05, # Target entropy for continuous actions
}

# --- Neural Network Architecture ---
NeuralNet = {
    "hidden_layers_critic": [256, 256], # Hidden layer sizes for the critic networks
    "hidden_layers_actor": [256, 256]   # Hidden layer sizes for the actor network
}



