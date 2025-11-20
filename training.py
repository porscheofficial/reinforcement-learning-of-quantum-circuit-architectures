<<<<<<< Updated upstream
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
=======
import patch  
# Standard library
import os
from datetime import datetime
import shutil
import subprocess 
import argparse
import importlib.util
import logging
import time
import random
import numpy as np
import torch

# Local imports
from QuantumStateEnv import QuantumStateEnv
from SAC import SAC_Agent
from reward import Rewards


"""
Execution script and main training loop for the RL-based circuit framework.

This script:
  - loads a configuration module from a given path,
  - sets up result directories and logging,
  - initializes the quantum environment and SAC agent,
  - runs multiple training sessions and episodes,
  - stores per-episode and per-session results to disk.
"""


def load_config(config_path: str):
    """
    Dynamically load a config module from a given path.

    Args:
        config_path: Path to the Python configuration file.

    Returns:
        Imported config module.
    """
>>>>>>> Stashed changes
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None:
        raise FileNotFoundError(f"Could not find config file at {config_path}")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

<<<<<<< Updated upstream
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
=======

def main() -> None:
    """
    Main entry point for training.

    Parses command-line arguments, loads the configuration,
    sets up logging and directories, and executes the training loop.
    """
    # ------------------------------------------------------------------
    # Argument parsing and config loading
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Train RL-based quantum circuit architectures."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Name of the configuration file inside the 'config' folder.",
    )
    args = parser.parse_args()

    config_path = os.path.join("config", args.config)
    if not os.path.exists(config_path):
        print(f"Configuration file {config_path} does not exist.")
        return

    cfg = load_config(config_path)

    # ------------------------------------------------------------------
    # Result directories
    # ------------------------------------------------------------------
    results_folder = cfg.training["results_folder_name"]
    timestamp = datetime.now().isoformat()
    main_folder = os.path.join(results_folder, f"training_{timestamp}")

    if not os.path.exists(results_folder):
        os.makedirs(results_folder, exist_ok=True)
    if not os.path.exists(main_folder):
        os.makedirs(main_folder, exist_ok=True)

    # Save configuration for reproducibility
    text_file = os.path.join(main_folder, "config_info.txt")
    shutil.copy2(config_path, text_file)

    # ------------------------------------------------------------------
    # Logger setup
    # ------------------------------------------------------------------
    log_file = "Info.log"
    log_path = os.path.join(main_folder, log_file)
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("Info")

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("CUDA is available. The program is running on GPU.")
    else:
        device = torch.device("cpu")
        logger.info(
            "CUDA is not available. The program is running on CPU. "
            "This is not recommended."
        )

    # ------------------------------------------------------------------
    # Environment initialization
    # ------------------------------------------------------------------
    env = QuantumStateEnv(logger, cfg, main_folder, pred=False)

    # ------------------------------------------------------------------
    # Training sessions
    # ------------------------------------------------------------------
    for x in range(cfg.training["runs"]):
        logger.info("New training session is starting.")

        rwd = Rewards(cfg, device)
        agent = SAC_Agent(
            cfg,
            env,
            device,
            logger,
            model_path=None,
            session=None,
            pred=False,
        )

        # Parameters from config
>>>>>>> Stashed changes
        max_episodes = cfg.training["max_episodes"]
        max_gates = cfg.training["max_gates"]
        train_evaluate_ratio = cfg.training["train_evaluate_ratio"]
        random_episodes = cfg.training["random_episodes"]
<<<<<<< Updated upstream
        
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

=======

        results = []
        layer_scale = np.round(
            np.linspace(1 / max_gates, 1, max_gates), 3
        ).tolist()
        decay = max_episodes * max_gates
        bond_distance_range = np.arange(
            cfg.characteristics["start_bond_distance"],
            cfg.characteristics["end_bond_distance"],
            cfg.characteristics["step_size_bond_distance"],
        )

        # --------------------------------------------------------------
        # Single training session (over episodes)
        # --------------------------------------------------------------
        for e in range(max_episodes):
            # Randomly sample bond distance index for this episode
            index = random.randint(0, len(bond_distance_range) - 1)
            bond_distance = np.round(bond_distance_range[index], 2)

            # Evaluation episode if episode index is a multiple of the ratio
            evaluation_episode = (e % train_evaluate_ratio == 0)

            # Reset environment-specific variables for this episode
            (
                state,
                qustate,
                current_qucircuit,
                episode_circuit,
                episode_angles,
            ) = env.reset(index)
            done = False
            start_episodes = False
            E_before = cfg.characteristics["initial_energy"]

            # Initial episodes: actions are chosen randomly
            if e < random_episodes:
                start_episodes = True
                evaluation_episode = False

            # ----------------------------------------------------------
            # Episode loop: construct one circuit
            # ----------------------------------------------------------
            for i in range(max_gates):
                # Update target entropy schedule
                step = e * max_gates + i
                agent.update_target_entropy(step, decay)

                # Choose next action
                action, angle_action = agent.get_next_action(
                    state, evaluation_episode, start_episodes
                )

                # Track the circuit
                episode_circuit.append(action)
                episode_angles.append(angle_action)

                # Environment transition
                (
                    next_state,
                    qustate,
                    current_qucircuit,
                ) = env.step(
                    qustate,
                    action,
                    angle_action,
                    current_qucircuit,
                    i,
                    layer_scale,
                    index,
                )

                # Energy evaluation
                E = env.get_energy(qustate, index)

                # Dynamic reward updates
                rwd.dynamic_values_for_reward_update(E, index)
                agent.dynamic_values_for_reward_update(E, index)

                # Episode termination at last gate
                if i == max_gates - 1:
                    done = True
                    E_before = cfg.characteristics["initial_energy"]
                    episode_return = rwd.which_reward(
                        E, E_before, done, index, tensor_form=False
                    )

                # Store transition and update networks (only in training episodes)
                if not evaluation_episode:
                    agent.train_on_transition(
                        state,
                        action,
                        angle_action,
                        next_state,
                        E,
                        E_before,
                        done,
                        e,
                        index,
                    )

                E_before = E
                state = next_state

            # Console output for monitoring
            print(
                f"[run {x + 1}/{cfg.training['runs']}] "
                f"[episode {e + 1}/{max_episodes}] "
                f"energy = {E:.4f}  "
                f"return = {episode_return:.4f}  "
                f"bd = {bond_distance:.2f}",
                flush=True,
            )

            # Store per-episode results
            results.append(
                (
                    e,
                    E,
                    evaluation_episode,
                    episode_return,
                    episode_circuit,
                    episode_angles,
                    bond_distance,
                )
            )

        # --------------------------------------------------------------
        # Save all quantities of one training session
        # --------------------------------------------------------------
        session_folder = os.path.join(main_folder, f"training_session_{x}")
        if not os.path.exists(session_folder):
            os.makedirs(session_folder)

        rl_quantities = agent.rl_quantities()
        agent.save_policy(x, session_folder)

        results_file = os.path.join(session_folder, f"results_{x}.npy")
        rl_quantities_file = os.path.join(
            session_folder, f"rl_quantities_{x}.npy"
        )

        np.save(results_file, np.array(results, dtype=object))
        np.save(rl_quantities_file, np.array(rl_quantities, dtype=object))


>>>>>>> Stashed changes
if __name__ == "__main__":
    main()
