"""
Configuration for a Test Run.

This configuration uses a smaller number of episodes, gates, and runs for quick
testing and debugging purposes.
"""

# --- Training Parameters ---
training = {
    "max_episodes": 100,
    "max_gates": 4,
    "train_evaluate_ratio": 10,
    "random_episodes": 10,
    "runs": 1,
    "results_folder_name": "Test"
}

<<<<<<< Updated upstream
# --- System and Environment Characteristics ---
characteristics = {
    "system": "LiH4",
    "hf_start": "HF",
    "start_bond_distance": 2.0,
    "end_bond_distance": 2.2,
    "step_size_bond_distance": 0.1,
=======
characteristics= {
    "rl_state": 0,
    "system": "JSP",
    "ham_check": "off",
    "hf_start": "HF",
    "start_bond_distance": 2.0, 
    "end_bond_distance": 4.0, 
    "step_size_bond_distance": 1.0,
>>>>>>> Stashed changes
    "initial_energy": -7,
}

# --- Reward Function Parameters ---
reward = {
    "sigma_min": 0.02,
    "c_lin": 0.1,
    "c_exp": 5,
    "alpha": 0.1
}

# --- Gaussian Encoding for Bond Distance ---
gaussian_encoding = {
    "number_of_embeddings": 3,
    "start_interval": 1,
    "end_interval": 4
}

# --- Soft Actor-Critic (SAC) Hyperparameters ---
SACparam = {
    "batch_size": 32,
    "gamma": 1,
    "lr_critic": 0.001,
    "lr_actor": 0.001,
    "lr_alpha_d": 0.003,
    "lr_alpha_c": 0.003,
    "soft_update_factor": 0.005,
    "training_update_factor": 2,
    "target_entropy_deduction_d": 0.4,
    "target_entropy_deduction_c": 0.2,
}

# --- Replay Buffer Parameters ---
ReplayBuffer = {
    "capacity": 100000
}

# --- Neural Network Architecture ---
NeuralNet = {
    "hidden_layers_critic": [64, 64],
    "hidden_layers_actor": [64, 64]
}




















