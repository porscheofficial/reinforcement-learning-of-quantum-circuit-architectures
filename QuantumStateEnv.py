import cProfile
import numpy as np
from itertools import product
from qulacs import Observable, QuantumCircuit, QuantumState
from qulacs.gate import CZ, RX, RY, RZ, Identity
from scipy.optimize import minimize
from numpy import load
from hamiltonians.Ham_gen import Hamiltonian_generation
from hamiltonians.JSP import JSP_generation
import copy
import os
import tequila as tq
from typing import Dict, List, Tuple

"""
Environment for executing actions and returning energies and states
in the state-based representation.
"""


class QuantumStateEnv:
    """
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
