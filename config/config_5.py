training= {
    "max_episodes": 40000,
    "max_gates": 10,
    "train_evaluate_ratio": 10,
    "random_episodes": 100,
    "runs": 4,
    "results_folder_name": "H48_15"
}

characteristics= {
    "rl_state": 0,
    "system": "H48",
    "ham_check": "on",
    "hf_start": "HF",
    "start_bond_distance": 1.5, 
    "end_bond_distance": 1.6, 
    "step_size_bond_distance": 0.2,
    "initial_energy": 0,
}

reward= {
    "reward": "exp_moving",
    "mu_average": 50,
    "sigma_average": 70,
    "sigma_min": 0.01,
    "c_lin": 0.1,
    "c_exp": 5
}

gaussian_encoding= {
    "number_of_embeddings": 6,    
    "start_interval": 0.5,    
    "end_interval": 2.0
}

SACparam={
    "batch_size": 512,
    "gamma": 1,
    "lr_critic": 0.001,
    "lr_actor": 0.001,
    "lr_alpha_d": 0.003,
    "lr_alpha_c": 0.003,
    "decay_factor_c": 2.0,
    "decay_factor_d": 1.0,
    "soft_update_factor": 0.005,
    "training_update_factor": 50,
    "target_entropy_deduction_d": 0.4,
    "target_entropy_deduction_c": 0.2,
    "target_entropy_end_value_d": 1,
    "target_entropy_end_value_c": -1
}

NeuralNet={
    "hidden_layers_actor":  [256,128],
    "hidden_layers_critic":  [256,128,128]
}  

RBparam={
    "capacity": 50000
}
