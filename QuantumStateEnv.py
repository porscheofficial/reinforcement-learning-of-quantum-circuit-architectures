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
<<<<<<< Updated upstream
=======
import copy
import os
import tequila as tq
from typing import Dict, List, Tuple

"""
Environment for executing actions and returning energies and states
in the state-based representation.
"""
>>>>>>> Stashed changes


class QuantumStateEnv:
    """
<<<<<<< Updated upstream
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
=======
    Environment that:
      - generates and stores molecular or problem Hamiltonians,
      - maintains a discrete action dictionary,
      - applies gates to a Qulacs circuit and state,
      - returns a neural-network-compatible state representation and energies.
    """

    def __init__(self, logger, cfg, main_folder: str, pred: bool) -> None:
        self.cfg = cfg
        self.logger = logger

        # Hamiltonian generation objects
        ham_gen = Hamiltonian_generation(cfg, logger, main_folder)
        jsp_ham = JSP_generation(cfg, logger, main_folder)

        self.molecule = cfg.characteristics["system"]

        # Bond distance grid
        if pred:
            print("Hamiltonian generation can take up to a few minutes...")
            self.bond_distance_range = np.arange(
                cfg.characteristics["start_bond_distance"],
                cfg.characteristics["end_bond_distance"],
                0.01,
            )
        else:
            self.bond_distance_range = np.arange(
                cfg.characteristics["start_bond_distance"],
                cfg.characteristics["end_bond_distance"],
                cfg.characteristics["step_size_bond_distance"],
            )

        # Path for saving Hamiltonian data
        path = os.path.join(main_folder, "molecule_data.npy")

        # Generate Hamiltonians depending on the specified system
        if self.molecule in ("H48", "H48HF"):
            data = ham_gen.generate_Hamiltonian_H48(self.bond_distance_range)
            np.save(path, np.array(data, dtype=object))
        elif self.molecule in ("LiH4", "LiH6"):
            data = ham_gen.generate_Hamiltonian_LiH(self.bond_distance_range, self.molecule)
            np.save(path, np.array(data, dtype=object))
        elif self.molecule == "JSP":
            data = jsp_ham.JSP_ham(self.bond_distance_range)
            np.save(path, np.array(data, dtype=object))
        else:
            self.logger.warning(
                "Specified system is unknown. Supported systems: H48, LiH4, LiH6, JSP."
            )
            # Note: if an unknown system is specified, 'data' is not defined and
            # subsequent access will fail, consistent with the original behavior.

        # Extract Hamiltonians and bond distances
        self.H = [entry[0] for entry in data]
        self.bond_distances = [entry[2] for entry in data]

        # Embedding for bond distances (if more than one distance is present)
        if len(self.bond_distance_range) != 1:
            self.r_embedding = self.r_to_embedding(self.bond_distances)

        # Number of qubits inferred from the Hamiltonian dimension
        self.num_qubits = int(np.log2(np.shape(self.H[0])[0]))
        self.logger.info(f"Qubits: {self.num_qubits}")

        # Size of the neural-network state representation:
        #   2 * (2^num_qubits): real and imaginary amplitudes,
        # + 1                 : layer information,
        # + len(r_embedding[0]) if bond-distance embedding is used.
        if len(self.bond_distance_range) != 1:
            self.state_size = 2 * (2 ** self.num_qubits) + 1 + len(self.r_embedding[0])
        else:
            self.state_size = 2 * (2 ** self.num_qubits) + 1

        # Maximum number of gates per episode (used by the agent)
        self.max_gates = self.cfg.training["max_gates"]

        # Initial Qulacs state and its vector representation
        self.qustate = QuantumState(self.num_qubits)
        self.state = self.qustate.get_vector()

        # Action dictionary and number of available actions
        self.dictionary, self.action_size = self.dictionary()

    # -------------------------------------------------------------------------
    # Action dictionary
    # -------------------------------------------------------------------------
    def dictionary(self) -> Tuple[Dict[int, List[int]], int]:
        """
        Generate the dictionary of actions.

        Each action is encoded as:
            [control_qubit, target_qubit, rotation_qubit, rotation_axis]

        Rotation axis:
            1 -> RX, 2 -> RY, 3 -> RZ

        Conventions:
            - control_qubit == num_qubits    -> no CNOT gate
            - rotation_qubit == num_qubits   -> no rotation gate
            - both == num_qubits             -> identity operation
        """
        self.dictionary = dict()
        i = 0
        num_qubits = self.num_qubits

        # CNOT actions
        for c in range(num_qubits):
            for x in range(num_qubits):
                if c != x:
                    # CNOT(c, x)
                    self.dictionary[i] = [c, x, num_qubits, 0]
                    i += 1

        # Single-qubit rotation actions (RX, RY, RZ)
        for r, h in product(range(num_qubits), range(1, 4)):
            self.dictionary[i] = [num_qubits, 0, r, h]
            i += 1

        number_of_actions = i - 1

        return self.dictionary, number_of_actions

    # -------------------------------------------------------------------------
    # Step: apply action and build NN state
    # -------------------------------------------------------------------------
    def step(
        self,
        qustate: QuantumState,
        chosed_action: int,
        angle_action: float,
        current_qucircuit: QuantumCircuit,
        i: int,
        layer_scale,
        index: int,
    ):
        """
        Apply the chosen action to the circuit and state, and return the
        updated neural-network state representation.

        Args:
            qustate: Current Qulacs state.
            chosed_action: Index of the action in the action dictionary.
            angle_action: Continuous parameter associated with the action.
            current_qucircuit: Current Qulacs circuit.
            i: Current layer index (time step within the episode).
            layer_scale: Array encoding the layer index as a scalar feature.
            index: Index of the current bond distance.

        Returns:
            outputstate: State representation as 1D numpy array.
            qustate: Updated Qulacs state.
            current_qucircuit: Updated Qulacs circuit.
        """
        action = self.dictionary[chosed_action]

        # Apply CNOT gate if a control qubit is specified
        if action[0] != self.num_qubits:
            current_qucircuit.add_CNOT_gate(action[0], action[1])

        # Apply single-qubit rotation if a rotation qubit is specified
        elif action[2] != self.num_qubits:
            angle = [angle_action]
            if action[3] == 1:
                current_qucircuit.add_RX_gate(action[2], angle[0])
            elif action[3] == 2:
                current_qucircuit.add_RY_gate(action[2], angle[0])
            elif action[3] == 3:
                current_qucircuit.add_RZ_gate(action[2], angle[0])

        # Update Qulacs state from the circuit
        qustate = QuantumState(self.num_qubits)
        current_qucircuit.update_quantum_state(qustate)

        # Quantum state representation for the neural network
        state = qustate.get_vector()
        nnstate = np.stack([np.real(state), np.imag(state)]).flatten()

        # Add layer information
        outputstate = np.concatenate((nnstate, [layer_scale[i]]))

        # Add bond-distance embedding if present
        if len(self.bond_distance_range) != 1:
            outputstate = np.concatenate((outputstate, self.r_embedding[index]))

        return outputstate, qustate, current_qucircuit

    # -------------------------------------------------------------------------
    # Energy computation
    # -------------------------------------------------------------------------
    def get_energy(self, qustate: QuantumState, index: int) -> float:
        """
        Compute the energy ⟨ψ|H|ψ⟩ for the given bond-distance index.

        Args:
            qustate: Qulacs state.
            index: Index of the Hamiltonian corresponding to the bond distance.

        Returns:
            Real-valued energy expectation.
        """
        state = qustate.get_vector()
        E = np.real(np.vdot(state, np.dot(self.H[index], state)))
        return E

    # -------------------------------------------------------------------------
    # Bond-distance embedding
    # -------------------------------------------------------------------------
    def r_to_embedding(self, r):
        """
        Compute Gaussian embeddings for a list or array of bond distances.

        Args:
            r: Iterable of bond distances.

        Returns:
            Array of shape (len(r), n_embeddings) containing the embeddings.
        """
        r = np.array(r)
        n = self.cfg.gaussian_encoding["number_of_embeddings"]
        a = self.cfg.gaussian_encoding["start_interval"]
        b = self.cfg.gaussian_encoding["end_interval"]

        # Means of Gaussian basis functions
        mu_k = np.linspace(a, b, n)

        # Shared standard deviation for all basis functions
        sigma = (b - a) / n

        return np.exp(-0.5 * ((r[:, np.newaxis] - mu_k) / sigma) ** 2)

    # -------------------------------------------------------------------------
    # Episode reset
    # -------------------------------------------------------------------------
    def reset(self, index: int):
        """
        Reset the environment for a new episode.

        The initial circuit is prepared in a Hartree–Fock (HF) or weakly
        stretched (WS) reference state, depending on configuration.

        Args:
            index: Index of the current bond distance.

        Returns:
            outputstate: Initial state representation as 1D numpy array.
            qustate: Initial Qulacs state.
            current_qucircuit: Initial Qulacs circuit.
            current_circuit: Empty list for storing discrete actions.
            angles: Empty list for storing continuous angles.
        """
        current_qucircuit = QuantumCircuit(self.num_qubits)

        hf_start = self.cfg.characteristics["hf_start"]
        system = self.cfg.characteristics["system"]

        # Prepare reference state depending on system and starting configuration
        if hf_start == "HF" and system == "LiH4":
            current_qucircuit.add_X_gate(1)
            current_qucircuit.add_X_gate(0)
        elif hf_start == "WS" and system == "LiH4":
            current_qucircuit.add_X_gate(3)
            current_qucircuit.add_X_gate(2)
        elif hf_start == "HF" and system == "LiH6":
            current_qucircuit.add_X_gate(0)
            current_qucircuit.add_X_gate(3)
        elif hf_start == "WS" and system == "LiH6":
            current_qucircuit.add_X_gate(2)
            current_qucircuit.add_X_gate(5)
        elif hf_start == "HF" and system == "H48":
            current_qucircuit.add_X_gate(7)
            current_qucircuit.add_X_gate(6)
            current_qucircuit.add_X_gate(3)
            current_qucircuit.add_X_gate(2)
        elif hf_start == "WS" and system == "H48":
            current_qucircuit.add_X_gate(0)
            current_qucircuit.add_X_gate(1)
            current_qucircuit.add_X_gate(5)
            current_qucircuit.add_X_gate(4)

        # Update Qulacs state from the initial circuit
        qustate = QuantumState(self.num_qubits)
        current_qucircuit.update_quantum_state(qustate)

        state = qustate.get_vector()
        current_circuit: List[int] = []
        angles: List[float] = []

        # Initial state representation
        nnstate = np.stack([np.real(state), np.imag(state)]).flatten()
        outputstate = np.concatenate((nnstate, [-1]))  # initial layer indicator

        if len(self.bond_distance_range) != 1:
            outputstate = np.concatenate((outputstate, self.r_embedding[index]))

        return outputstate, qustate, current_qucircuit, current_circuit, angles
>>>>>>> Stashed changes
