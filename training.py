"""
This script serves as the main execution file for training the reinforcement
learning agent to design quantum circuits.
"""
import argparse
import importlib.util
import logging
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch

import patch  # Apply the patch for qiskit_algorithms
from QuantumStateEnv import QuantumStateEnv
from SAC import SAC_Agent


def load_config(config_path: str):
    """Loads a configuration module from a given path."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None:
        raise FileNotFoundError(f"Could not find config file at {config_path}")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def setup_logging(main_folder: str) -> logging.Logger:
    """Sets up the logger for the training session."""
    log_file = 'Info.log'
    log_path = os.path.join(main_folder, log_file)
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('TrainingLogger')
    return logger

def main():
    """Main function to run the training loop."""
    parser = argparse.ArgumentParser(description="Train a SAC agent to design quantum circuits.")
    parser.add_argument('config', type=str, help='Path to the configuration file (e.g., config_test.py).')
    args = parser.parse_args()

    config_path = os.path.join('config', args.config)
    cfg = load_config(config_path)

    # --- Setup Folders and Logging ---
    results_folder = cfg.training["results_folder_name"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    main_folder = os.path.join(results_folder, f"training_{timestamp}")
    os.makedirs(main_folder, exist_ok=True)
    
    shutil.copy2(config_path, os.path.join(main_folder, "config_info.txt"))
    logger = setup_logging(main_folder)

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == 'cpu':
        logger.warning("CUDA not available. Training on CPU is not recommended.")

    # --- Environment and Agent Initialization ---
    env = QuantumStateEnv(logger, cfg, main_folder, pred=False)

    for session_num in range(cfg.training["runs"]):
        logger.info(f"--- Starting Training Session {session_num + 1}/{cfg.training['runs']} ---")
        
        agent = SAC_Agent(cfg, env, device, logger, model_path=main_folder, session=session_num, pred=False)

        # --- Training Parameters ---
        max_episodes = cfg.training["max_episodes"]
        max_gates = cfg.training["max_gates"]
        train_evaluate_ratio = cfg.training["train_evaluate_ratio"]
        random_episodes = cfg.training["random_episodes"]
        
        layer_scale = [-1.0] + list(np.linspace(-1, 1, max_gates + 1))
        bond_distance_range = np.arange(
            cfg.characteristics["start_bond_distance"],
            cfg.characteristics["end_bond_distance"],
            cfg.characteristics["step_size_bond_distance"]
        )

        # --- Main Training Loop ---
        for e in range(max_episodes):
            # Randomly sample a bond distance for this episode
            index = random.randint(0, len(bond_distance_range) - 1)
            
            # Determine if it's an evaluation or random exploration episode
            evaluation_episode = e % train_evaluate_ratio == 0
            start_episodes = e < random_episodes
            if start_episodes:
                evaluation_episode = False

            # Reset environment for the new episode
            state, qustate, current_qucircuit, episode_circuit, episode_angles = env.reset(index)
            E_before = env.get_energy(qustate, index)
            
            # --- Episode Loop ---
            for i in range(max_gates):
                action, angle_action = agent.get_next_action(state, evaluation_episode, start_episodes)

                episode_circuit.append(action)
                episode_angles.append(angle_action)

                next_state, qustate, current_qucircuit = env.step(
                    qustate, action, angle_action, current_qucircuit, i, layer_scale, index
                )
                
                energy = env.get_energy(qustate, index)
                done = (i == max_gates - 1)

                if not evaluation_episode:
                    agent.train_on_transition(state, action, angle_action, next_state, energy, E_before, done, e, index)
                
                E_before = energy
                state = next_state

            # --- Logging and Output ---
            if evaluation_episode:
                final_energy = E_before
                print(
                    f"Session: {session_num + 1}/{cfg.training['runs']}, "
                    f"Eval Episode: {e}/{max_episodes}, "
                    f"Bond Distance: {bond_distance_range[index]:.2f}, "
                    f"Final Energy: {final_energy:.6f}"
                )
        
        # Save models at the end of the session
        agent.save_models()
        logger.info(f"--- Training Session {session_num + 1} Complete ---")

if __name__ == "__main__":
    main()
