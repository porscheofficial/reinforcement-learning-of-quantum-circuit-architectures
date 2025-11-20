"""
This script runs the prediction loop for the reinforcement learning agent to find
optimal quantum circuit architectures for specific molecular configurations.

It loads a trained agent, iterates through a range of bond distances, and for each
distance, it predicts the best quantum circuit architecture.
"""
# Standard
import os
import argparse
import importlib.util
import logging
<<<<<<< Updated upstream
import numpy as np
import torch

from QuantumStateEnv import QuantumStateEnv
from SAC import SAC_Agent
import config as cfg

def load_config(config_path: str):
    """
    Loads a configuration module from the given file path.

    Args:
        config_path: The path to the Python configuration file.

    Returns:
        The loaded configuration module.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    
=======
import time
import random

# Third-party
import numpy as np
import torch

# Local imports
from QuantumStateEnv import QuantumStateEnv
from SAC import SAC_Agent
from reward import Rewards
import config as cfg  # kept for compatibility, even though it is shadowed later


"""
Prediction script for evaluating trained RL-based quantum circuit architectures.

This script:
  - loads a configuration module from a given path,
  - initializes the environment in prediction mode,
  - loads trained actor models for each training session,
  - predicts circuits and energies on a dense bond-distance grid,
  - saves the resulting predictions to disk.
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
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module

<<<<<<< Updated upstream
def setup_logging() -> logging.Logger:
    """
    Sets up a null logger to avoid logging messages during prediction.

    Returns:
        A logger instance with a NullHandler.
    """
    logger = logging.getLogger('prediction_logger')
    logger.addHandler(logging.NullHandler())
    return logger

def main():
    """
    Main function to run the prediction process.
    """
    parser = argparse.ArgumentParser(description="Predict quantum circuit architectures for a given configuration.")
    parser.add_argument('config', type=str, help='The name of the configuration file (e.g., config_test.py).')
    parser.add_argument('path', type=str, help='Path to the directory containing the trained models.')
    args = parser.parse_args()

    config_path = os.path.join('config', args.config)
    model_path = args.path
    
    cfg = load_config(config_path)
    logger = setup_logging()
    main_folder = '.'

    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == 'cpu':
        logger.warning('CUDA not available. Running on CPU is not recommended for performance.')

    # Initialize the quantum environment
    env = QuantumStateEnv(logger, cfg, main_folder, pred=True)
    max_gates = cfg.training["max_gates"]
    layer_scale = np.round(np.linspace(1 / max_gates, 1, max_gates), 3).tolist()

    # Define the range of bond distances to investigate
    bond_distance_range = np.arange(
        cfg.characteristics["start_bond_distance"],
        cfg.characteristics["end_bond_distance"],
        0.01
    )

    # Run prediction for each session/run defined in the config
    for session in range(cfg.training["runs"]):
        print(f"--- Starting Prediction for Session {session} ---")
        agent = SAC_Agent(cfg, env, device, logger, model_path, session, pred=True)
        results = []

        # Loop over each bond distance
        for bond_idx, bond_distance in enumerate(bond_distance_range):
            bond_distance = np.round(bond_distance, 2)
            print(f"Predicting for bond distance: {bond_distance} Å")

            # Reset environment for the new episode
            state, _, _, episode_circuit, episode_angles = env.reset(bond_idx)
            
            # Prediction loop for a single bond distance
            for i in range(max_gates):
                # Get the next action from the agent deterministically
                action, angle_action = agent.get_next_action(state, evaluation_episode=True, start_episodes=False)

                # Record the chosen action and angle
                episode_circuit.append(action)
                episode_angles.append(angle_action)

                # Take a step in the environment
                next_state, _, _ = env.step(
                    None, action, angle_action, None, i, layer_scale, bond_idx
                )
                state = next_state

                if i == max_gates - 1:
                    print(f"  -> Final circuit for {bond_distance} Å predicted.")
                    # Here you would typically save or process the final circuit
                    # For example: results.append((bond_distance, episode_circuit, episode_angles))
        
        print(f"--- Session {session} Complete ---")
        # Process or save results for the session here

=======

def main() -> None:
    """
    Main entry point for prediction.

    Parses command-line arguments, loads the configuration, initializes
    the environment in prediction mode, and generates predictions for
    each trained session and bond distance.
    """
    # ------------------------------------------------------------------
    # Argument parsing and config loading
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Predict energies and circuits using trained RL models."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Name of the configuration file inside the 'config' folder (e.g. my_config.py).",
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the directory containing the trained models.",
    )
    args = parser.parse_args()

    config_path = os.path.join("config", args.config)
    model_path = args.path

    if not os.path.exists(config_path):
        print(f"Configuration file {config_path} does not exist.")
        return

    cfg = load_config(config_path)

    # ------------------------------------------------------------------
    # Logger and device
    # ------------------------------------------------------------------
    logger = logging.getLogger("dummy")
    logger.addHandler(logging.NullHandler())
    main_folder = "."

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("CUDA is available. The program is running on GPU.")
    else:
        device = torch.device("cpu")
        logger.info(
            "CUDA is not available. The program is running on CPU. This is not recommended."
        )

    # ------------------------------------------------------------------
    # Environment in prediction mode
    # ------------------------------------------------------------------
    env = QuantumStateEnv(logger, cfg, main_folder, pred=True)

    max_gates = cfg.training["max_gates"]
    layer_scale = np.round(
        np.linspace(1 / max_gates, 1, max_gates), 3
    ).tolist()

    # Prediction with step size 0.01 over the interval of interest
    bond_distance_range = np.arange(
        cfg.characteristics["start_bond_distance"],
        cfg.characteristics["end_bond_distance"],
        0.01,
    )

    # ------------------------------------------------------------------
    # Loop over training sessions
    # ------------------------------------------------------------------
    for session in range(0, cfg.training["runs"]):
        agent = SAC_Agent(
            cfg,
            env,
            device,
            logger,
            model_path,
            session,
            pred=True,
        )
        results = []

        # --------------------------------------------------------------
        # Loop over bond distances for this session
        # --------------------------------------------------------------
        for k in range(len(bond_distance_range)):
            bond_distance = np.round(bond_distance_range[k], 2)

            # Prediction-only: no further training, deterministic evaluation
            evaluation_episode = True

            # Reset environment for this parameter
            (
                state,
                qustate,
                current_qucircuit,
                episode_circuit,
                episode_angles,
            ) = env.reset(k)
            done = False
            start_episodes = False
            E_before = cfg.characteristics["initial_energy"]

            # ----------------------------------------------------------
            # Predict circuit for one parameter value
            # ----------------------------------------------------------
            for i in range(max_gates):
                # Choose next action (deterministic in evaluation mode)
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
                    k,
                )

                # Final energy evaluation at last gate
                if i == max_gates - 1:
                    done = True
                    E = env.get_energy(qustate, k)

                state = next_state

            # Console output for monitoring
            print(
                "session:",
                session,
                "parameter:",
                np.round(bond_distance, 2),
                "Predicted energy:",
                np.round(E, 4),
                flush=True,
            )

            # Store per-parameter result
            results.append((E, episode_circuit, episode_angles, bond_distance))

        # ------------------------------------------------------------------
        # Save predictions for this session
        # ------------------------------------------------------------------
        model_path_save = os.path.join(model_path, f"training_session_{session}")
        os.makedirs(model_path_save, exist_ok=True)

        np.save(
            os.path.join(model_path_save, f"unseen_prediction_{session}.npy"),
            results,
        )


>>>>>>> Stashed changes
if __name__ == "__main__":
    main()
