"""
Configuration for Training Run 6: H48 Molecule, wide bond distance scan (0.5-1.6 Å).
"""

# --- Training Parameters ---
training = {
    "max_episodes": 50000,
    "max_gates": 10,
    "train_evaluate_ratio": 10,
    "random_episodes": 100,
    "runs": 4,
    "results_folder_name": "H48_T"
}

# --- System and Environment Characteristics ---
characteristics = {
    "system": "H48",
    "hf_start": "HF",
    "start_bond_distance": 0.5,
    "end_bond_distance": 1.6,
    "step_size_bond_distance": 0.1,
    "initial_energy": 0,
}

# --- Reward Function Parameters ---
reward = {
    "sigma_min": 0.0075,
    "c_lin": 0.1,
    "c_exp": 5,
    "alpha": 0.1
}

# --- Gaussian Encoding for Bond Distance ---
gaussian_encoding = {
    "number_of_embeddings": 6,
    "start_interval": 0.5,
    "end_interval": 2.0
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
    "target_entropy_deduction_d": 0.4,
    "target_entropy_deduction_c": 0.2,
}

# --- Neural Network Architecture ---
NeuralNet = {
    "hidden_layers_critic": [256, 256],
    "hidden_layers_actor": [256, 256]
}




