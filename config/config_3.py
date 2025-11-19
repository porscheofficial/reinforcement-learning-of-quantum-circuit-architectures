"""
Configuration for Training Run 3: LiH6 Molecule, 2.2-2.3 Å bond distance.
"""

# --- Training Parameters ---
training = {
    "max_episodes": 40000,
    "max_gates": 12,
    "train_evaluate_ratio": 10,
    "random_episodes": 100,
    "runs": 4,
    "results_folder_name": "LiH6_22"
}

# --- System and Environment Characteristics ---
characteristics = {
    "system": "LiH6",
    "hf_start": "WS",
    "start_bond_distance": 2.2,
    "end_bond_distance": 2.3,
    "step_size_bond_distance": 0.2,
    "initial_energy": -7,
}

# --- Reward Function Parameters ---
reward = {
    "sigma_min": 0.005,
    "c_lin": 0.1,
    "c_exp": 5,
    "alpha": 0.1
}

# --- Gaussian Encoding for Bond Distance ---
gaussian_encoding = {
    "number_of_embeddings": 6,
    "start_interval": 1,
    "end_interval": 4
}

# --- Soft Actor-Critic (SAC) Hyperparameters ---
SACparam = {
    "batch_size": 512,
    "gamma": 1,
    "lr_critic": 0.001,
    "lr_actor": 0.001,
    "lr_alpha_d": 0.003,
    "lr_alpha_c": 0.003,
    "soft_update_factor": 0.005,
    "training_update_factor": 50,
    "target_entropy_deduction_d": 0.3,
    "target_entropy_deduction_c": 0.1,
}

# --- Neural Network Architecture ---
NeuralNet = {
    "hidden_layers_critic": [256, 256],
    "hidden_layers_actor": [256, 256]
}
