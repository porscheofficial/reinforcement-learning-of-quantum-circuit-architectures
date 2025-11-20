# Standard
import os
from datetime import datetime
import shutil
import subprocess
import argparse
import importlib.util
import logging
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
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


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


if __name__ == "__main__":
    main()
