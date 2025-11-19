"""
This module defines the QuantumStateEnv class, which provides an environment for
the reinforcement learning agent to interact with a quantum system.

The environment handles the simulation of a quantum state, application of quantum gates
(actions), and calculation of energies (rewards).
"""
import logging
import os
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
from qulacs import QuantumCircuit, QuantumState

from hamiltonians.Ham_gen import Hamiltonian_generation
from hamiltonians.JSP import JSP_generation


class QuantumStateEnv:
    """
    An environment that represents the state of a quantum system and allows an RL
    agent to modify it by applying quantum gates.
    """

    def __init__(self, logger: logging.Logger, cfg: Any, main_folder: str, pred: bool = False):
        """
        Initializes the quantum environment.

        Args:
            logger: A logger for recording information.
            cfg: A configuration object containing system and training parameters.
            main_folder: The root directory for data files.
            pred: A flag indicating whether the environment is for prediction.
        """
        self.logger = logger
        self.cfg = cfg
        self.num_qubits = 0
        self.max_gates = self.cfg.training["max_gates"]

        self.hamiltonians, self.bond_distances = self._load_or_generate_hamiltonians(main_folder, pred)
        self.num_qubits = int(np.log2(self.hamiltonians[0].shape[0]))
        self.logger.info(f"System initialized with {self.num_qubits} qubits.")

        self.r_embedding = self._create_gaussian_embedding(self.bond_distances) if len(self.bond_distances) > 1 else None
        
        self.state_size = self._calculate_state_size()
        self.action_space, self.action_size = self._create_action_dictionary()

    def _load_or_generate_hamiltonians(self, main_folder: str, pred: bool) -> Tuple[List[np.ndarray], List[float]]:
        """Loads or generates the Hamiltonians for the specified molecule and bond distances."""
        hamiltonian_path = os.path.join(main_folder, "molecule_data.npy")
        molecule = self.cfg.characteristics["system"]
        
        if pred:
            print("Hamiltonian generation can take up to a few minutes...")
            bond_distance_range = np.arange(
                self.cfg.characteristics["start_bond_distance"],
                self.cfg.characteristics["end_bond_distance"],
                0.01
            )
        else:
            bond_distance_range = np.arange(
                self.cfg.characteristics["start_bond_distance"],
                self.cfg.characteristics["end_bond_distance"],
                self.cfg.characteristics["step_size_bond_distance"]
            )

        if molecule in ["H48", "H48HF", "LiH4", "LiH6"]:
            ham_gen = Hamiltonian_generation(self.cfg, self.logger, main_folder)
            if "H4" in molecule:
                data = ham_gen.generate_Hamiltonian_H4(bond_distance_range)
            else: # LiH
                data = ham_gen.generate_Hamiltonian_LiH(bond_distance_range, molecule)
        elif molecule == "JSP":
            jsp_ham_gen = JSP_generation(self.cfg, self.logger, main_folder)
            data = jsp_ham_gen.JSP_ham(bond_distance_range)
        else:
            self.logger.warning(f"Specified system '{molecule}' is unknown.")
            raise ValueError(f"Unknown system: {molecule}")
        
        np.save(hamiltonian_path, np.array(data, dtype=object))
        
        hamiltonians = [entry[0] for entry in data]
        bond_distances = [entry[2] for entry in data]
        return hamiltonians, bond_distances

    def _calculate_state_size(self) -> int:
        """Calculates the size of the state representation for the neural network."""
        base_state_size = 2 * (2**self.num_qubits) + 1  # Real and imaginary parts + layer info
        if self.r_embedding is not None:
            return base_state_size + len(self.r_embedding[0])
        return base_state_size

    def _create_action_dictionary(self) -> Tuple[Dict[int, List[int]], int]:
        """
        Generates the dictionary of possible actions (quantum gates).

        Returns:
            A tuple containing the action dictionary and the number of actions.
        """
        action_dict = {}
        action_id = 0

        # CNOT gates
        for c in range(self.num_qubits):
            for x in range(self.num_qubits):
                if c != x:
                    action_dict[action_id] = [c, x, self.num_qubits, 0]  # [ctrl, target, rot_q, rot_axis]
                    action_id += 1
        
        # Rotation gates (RX, RY, RZ)
        for r, h in product(range(self.num_qubits), range(1, 4)):
            action_dict[action_id] = [self.num_qubits, 0, r, h]
            action_id += 1
            
        return action_dict, action_id

    def step(self, quantum_state: QuantumState, action: int, angle: float,
             circuit: QuantumCircuit, layer_index: int, layer_scale: List[float],
             bond_distance_index: int) -> Tuple[np.ndarray, QuantumState, QuantumCircuit]:
        """
        Applies a gate to the quantum state and returns the new state representation.
        """
        gate_params = self.action_space[action]
        control_qubit, target_qubit, rotation_qubit, rotation_axis = gate_params

        if control_qubit != self.num_qubits:  # It's a CNOT gate
            circuit.add_CNOT_gate(control_qubit, target_qubit)
        elif rotation_qubit != self.num_qubits:  # It's a rotation gate
            if rotation_axis == 1:
                circuit.add_RX_gate(rotation_qubit, angle)
            elif rotation_axis == 2:
                circuit.add_RY_gate(rotation_qubit, angle)
            elif rotation_axis == 3:
                circuit.add_RZ_gate(rotation_qubit, angle)
        
        # Update the quantum state by applying the new circuit
        new_quantum_state = QuantumState(self.num_qubits)
        circuit.update_quantum_state(new_quantum_state)

        # Create the state representation for the neural network
        state_vector = new_quantum_state.get_vector()
        nn_state = np.concatenate([np.real(state_vector), np.imag(state_vector)])
        
        output_state = np.concatenate([nn_state, [layer_scale[layer_index]]])
        if self.r_embedding is not None:
            output_state = np.concatenate([output_state, self.r_embedding[bond_distance_index]])

        return output_state, new_quantum_state, circuit

    def get_energy(self, quantum_state: QuantumState, index: int) -> float:
        """Calculates the expectation value (energy) of the quantum state."""
        state_vector = quantum_state.get_vector()
        energy = np.real(np.vdot(state_vector, np.dot(self.hamiltonians[index], state_vector)))
        return energy

    def _create_gaussian_embedding(self, r_values: List[float]) -> np.ndarray:
        """Creates a Gaussian embedding for the bond distances."""
        r = np.array(r_values)
        n = self.cfg.gaussian_encoding["number_of_embeddings"]
        a = self.cfg.gaussian_encoding["start_interval"]
        b = self.cfg.gaussian_encoding["end_interval"]
        
        mu_k = np.linspace(a, b, n)
        sigma = (b - a) / n
        
        return np.exp(-0.5 * ((r[:, np.newaxis] - mu_k) / sigma) ** 2)

    def _get_initial_circuit(self) -> QuantumCircuit:
        """Creates the initial quantum circuit based on the configuration."""
        circuit = QuantumCircuit(self.num_qubits)
        hf_start = self.cfg.characteristics.get("hf_start")
        system = self.cfg.characteristics.get("system")

        initial_states = {
            ("HF", "LiH4"): [(1, True), (0, True)],
            ("WS", "LiH4"): [(3, True), (2, True)],
            ("HF", "LiH6"): [(0, True), (3, True)],
            ("WS", "LiH6"): [(2, True), (5, True)],
            ("HF", "H48"): [(7, True), (6, True), (3, True), (2, True)],
            ("WS", "H48"): [(0, True), (1, True), (5, True), (4, True)],
        }

        gates_to_add = initial_states.get((hf_start, system), [])
        for qubit, is_x_gate in gates_to_add:
            if is_x_gate:
                circuit.add_X_gate(qubit)
        
        return circuit

    def reset(self, index: int) -> Tuple[np.ndarray, QuantumState, QuantumCircuit, List, List]:
        """
        Resets the environment to an initial state for a new episode.
        """
        initial_circuit = self._get_initial_circuit()
        
        quantum_state = QuantumState(self.num_qubits)
        initial_circuit.update_quantum_state(quantum_state)
        
        state_vector = quantum_state.get_vector()
        
        # Initialize state representation for the neural network
        nn_state = np.concatenate([np.real(state_vector), np.imag(state_vector)])
        
        # Use a consistent layer scaling for the initial state
        layer_scale = [-1.0] + list(np.linspace(-1, 1, self.max_gates + 1))
        
        output_state = np.concatenate([nn_state, [layer_scale[0]]])
        if self.r_embedding is not None:
            output_state = np.concatenate([output_state, self.r_embedding[index]])
     
        return output_state, quantum_state, initial_circuit, [], []
