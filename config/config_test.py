training= {
    "max_episodes": 10000,
    "max_gates": 4,
    "train_evaluate_ratio": 10,
    "random_episodes": 50,
    "runs": 2,
    "results_folder_name": "Test"
}

characteristics= {
    "rl_state": 0,
    "system": "JSP",
    "ham_check": "off",
    "hf_start": "HF",
    "start_bond_distance": 2.0, 
    "end_bond_distance": 4.0, 
    "step_size_bond_distance": 1.0,
    "initial_energy": -7,
}

reward= {
    "reward": "exp_moving",
    "mu_average": 30,
    "sigma_average": 50,
    "sigma_min": 0.02,
    "c_lin": 0.1,
    "c_exp": 5
}

gaussian_encoding= {
    "number_of_embeddings": 3,    
    "start_interval": 1 ,    
    "end_interval": 4
}

SACparam={
    "batch_size": 10,
    "gamma": 1,
    "lr_critic": 0.001,
    "lr_actor": 0.001,
    "lr_alpha_d": 0.003,
    "lr_alpha_c": 0.003,
    "decay_factor_c": 2,
    "decay_factor_d": 1,
    "soft_update_factor": 0.005,
    "training_update_factor": 1,
    "target_entropy_deduction_d": 0.4,
    "target_entropy_deduction_c": 0.2,
    "target_entropy_end_value_d": 0.5,
    "target_entropy_end_value_c": -2
}

NeuralNet={
    "hidden_layers_actor":  [256,128],
    "hidden_layers_critic":  [256,128,128]
}  

RBparam={
    "capacity": 10000
}




















