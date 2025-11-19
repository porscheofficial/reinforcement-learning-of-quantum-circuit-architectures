"""
Configuration for Training Run 7: JSP Problem.
"""

# --- Training Parameters ---
training = {
    "max_episodes": 45000,
    "max_gates": 12,
    "train_evaluate_ratio": 10,
    "random_episodes": 100,
    "runs": 4,
    "results_folder_name": "JSP"
}

# --- System and Environment Characteristics ---
characteristics = {
    "system": "JSP",
    "hf_start": "0",
    "start_bond_distance": 1,
    "end_bond_distance": 3.5,
    "step_size_bond_distance": 1,
    "initial_energy": 30,
    "gate_set": "HEA"
}

# --- Reward Function Parameters ---
reward = {
    "sigma_min": 0.25,
    "c_lin": 1,
    "c_exp": 5,
    "alpha": 0.1
}

# --- Gaussian Encoding for Bond Distance ---
gaussian_encoding = {
    "number_of_embeddings": 3,
    "start_interval": 0.9,
    "end_interval": 3.1
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
    "target_entropy_deduction_d": 0.1,
    "target_entropy_deduction_c": 0.05,
}

# --- Neural Network Architecture ---
NeuralNet = {
    "hidden_layers_critic": [256, 256],
    "hidden_layers_actor": [256, 256]
}

