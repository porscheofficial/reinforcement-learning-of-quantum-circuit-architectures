"""
This module handles the generation of the Hamiltonian for the Job Shop Scheduling
Problem (JSP).
"""
import logging
from typing import Any, List, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp


class JSP_generation:
    """
<<<<<<< Updated upstream
    A class to generate the Hamiltonian for a specific instance of the Job Shop
    Scheduling Problem.
    """

    def __init__(self, cfg: Any, logger: logging.Logger, main_folder: str):
        """
        Initializes the JSP_generation class.

        Args:
            cfg: A configuration object.
            logger: A logger for recording information.
            main_folder: The root directory for data files.
=======
    Generation of Hamiltonians for a small Job-Shop Scheduling Problem (JSP).

    The encoding used here follows a binary-variable formulation where
    each variable is represented by a qubit projector (I - Z) / 2. The
    Hamiltonian consists of:
      - constraint penalty terms enforcing scheduling consistency, and
      - an optimization term penalizing makespan.

    The construction is fixed to:
      - N = 3 jobs,
      - m = 2 machines,
      - M = 1 (max runtime difference),
      - with job lengths L = [1, 1, L3[q]] for each element in L3.
    """

    def __init__(self, cfg, logger, main_folder):
        """
        Args:
            cfg: Configuration object/module (kept for consistency).
            logger: Logger instance for diagnostics.
            main_folder: Path to main folder (not used explicitly here,
                         kept for interface symmetry with other generators).
>>>>>>> Stashed changes
        """
        self.cfg = cfg
        self.logger = logger
        self.main_folder = main_folder
<<<<<<< Updated upstream

    def generate_jsp_hamiltonian(self, L3_values: List[float]) -> List[Tuple[np.ndarray, float, float]]:
        """
        Generates the Hamiltonian for the JSP for a given set of L3 values.

        This specific instance corresponds to a 3-job, 2-machine problem.

        Args:
            L3_values: A list of values for the length of the third job.

        Returns:
            A list of tuples, each containing the Hamiltonian matrix, its ground
            state energy, and the corresponding L3 value.
        """
        jsp_info = []

        for l3 in L3_values:
            job_lengths = [1, 1, l3]  # Lengths of the 3 jobs
            num_jobs = 3
            num_machines = 2
            max_runtime_diff = 1
            
            # Coefficients for the penalty and objective terms
            A_penalty = 4
            B_objective = 1
            
            num_qubits = num_jobs * num_machines + (num_machines - 1) * max_runtime_diff

            identity = SparsePauliOp("I" * num_qubits)
            pauli_ops = self._create_binary_variable_operators(num_qubits)

            # --- Construct the Penalty Hamiltonian (H_A) ---
            # Constraint 1: Each job must be assigned to exactly one machine.
            h_a1 = self._build_penalty_term1(num_jobs, num_machines, num_qubits, pauli_ops)
            
            # Constraint 2: Precedence constraints (not explicitly modeled here, assumed in problem structure)
            # Constraint 3: Capacity constraints (related to runtime differences)
            h_a2 = self._build_penalty_term2(job_lengths, num_jobs, num_machines, max_runtime_diff, num_qubits, pauli_ops)

            penalty_hamiltonian = h_a1 + h_a2

            # --- Construct the Objective Hamiltonian (H_B) ---
            objective_hamiltonian = self._build_objective_term(num_jobs, num_machines, num_qubits, pauli_ops)
=======

    def JSP_ham(self, L3):
        """
        Construct JSP Hamiltonians for a set of third-job lengths.

        For each value L3[q], a fixed JSP instance with
        L = [1, 1, L3[q]] is translated into a binary-variable Hamiltonian.

        Args:
            L3: Iterable of values specifying the processing time of
                the third job.

        Returns:
            List of tuples (H, ground_energy, L3_value), where:
              - H is the full Hamiltonian as a dense matrix,
              - ground_energy is the smallest eigenvalue of H,
              - L3_value is the corresponding job length rounded to 2 decimals.
        """
        JSP_info = []

        for q in range(len(L3)):
            # Fixed JSP instance parameters
            L = [1, 1, L3[q]]  # job lengths
            N = 3              # number of jobs
            m = 2              # number of machines
            M = 1              # max runtime difference
            A = 4              # penalty prefactor for constraints
            B = 1              # prefactor for the optimization term

            num_qubits = N * m + (m - 1) * M

            identity = SparsePauliOp.from_list(
                [("I" * num_qubits, 1)]
            ).to_matrix()
            operator_list = self.binary_variable_operators(num_qubits)

            dim = 2 ** num_qubits

            # --------------------------------------------------------------
            # First term: per-job machine allocation constraints
            # --------------------------------------------------------------
            first_term = np.zeros((dim, dim))
            for i in range(N):
                help_term_1 = np.zeros((dim, dim))
                for j in range(m):
                    help_term_1 = help_term_1 + operator_list[i + j * N].to_matrix()
                first_term = first_term + (identity - help_term_1) ** 2

            # --------------------------------------------------------------
            # Second term: timing/consistency constraints between machines
            # --------------------------------------------------------------
            second_term = np.zeros((dim, dim))
            for j in range(1, m):
                help_term_3 = np.zeros((dim, dim))
                for i in range(N):
                    help_term_3 = help_term_3 + L[i] * (
                        operator_list[i + j * N].to_matrix()
                        - operator_list[i].to_matrix()
                    )

                help_term_4 = np.zeros((dim, dim))
                for p in range(0, M):
                    help_term_4 = help_term_4 + (p + 1) * operator_list[
                        m * N + p + (j - 1) * M
                    ].to_matrix()

                second_term = second_term + (help_term_3 + help_term_4) ** 2

            # --------------------------------------------------------------
            # Optimization term: makespan contribution
            # --------------------------------------------------------------
            opt_term = np.zeros((dim, dim))
            for i in range(N):
                opt_term = opt_term + L[i] * operator_list[i].to_matrix()

            # Full Hamiltonian
            H = A * first_term + A * second_term + B * opt_term

            # Ground-state energy
            eigvals, eigvecs = LA.eigh(H)
            ground_energy = eigvals[0]

            JSP_info.append((H, ground_energy, np.round(L3[q], 2)))
>>>>>>> Stashed changes

            # --- Combine Hamiltonians ---
            total_hamiltonian = A_penalty * penalty_hamiltonian + B_objective * objective_hamiltonian
            
            hamiltonian_matrix = total_hamiltonian.to_matrix()
            
            # Calculate the ground state energy
            eigenvalues, _ = np.linalg.eigh(hamiltonian_matrix)
            ground_state_energy = np.min(eigenvalues)

<<<<<<< Updated upstream
            jsp_info.append((hamiltonian_matrix, ground_state_energy, l3))
            self.logger.info(f"Generated JSP Hamiltonian for L3 = {l3}, E0 = {ground_state_energy:.6f}")

        return jsp_info

    def _create_binary_variable_operators(self, num_qubits: int) -> List[SparsePauliOp]:
        """Creates Pauli Z operators to represent binary variables."""
        identity = SparsePauliOp("I" * num_qubits)
        operators = []
        for i in range(num_qubits):
            pauli_str = ["I"] * num_qubits
            pauli_str[i] = "Z"
            z_op = SparsePauliOp("".join(pauli_str))
            # Map eigenvalue -1 to 1 and 1 to 0
            operators.append(0.5 * (identity - z_op))
        return operators

    def _build_penalty_term1(self, num_jobs, num_machines, num_qubits, pauli_ops):
        """Builds the first penalty term of the JSP Hamiltonian."""
        penalty_term = SparsePauliOp("I" * num_qubits, 0)
        identity = SparsePauliOp("I" * num_qubits)
        for i in range(num_jobs):
            sum_term = SparsePauliOp("I" * num_qubits, 0)
            for j in range(num_machines):
                sum_term += pauli_ops[i + j * num_jobs]
            penalty_term += (identity - sum_term) ** 2
        return penalty_term

    def _build_penalty_term2(self, job_lengths, num_jobs, num_machines, max_runtime_diff, num_qubits, pauli_ops):
        """Builds the second penalty term for machine capacity constraints."""
        penalty_term = SparsePauliOp("I" * num_qubits, 0)
        for j in range(1, num_machines):
            sum_term1 = SparsePauliOp("I" * num_qubits, 0)
            for i in range(num_jobs):
                sum_term1 += job_lengths[i] * (pauli_ops[i + j * num_jobs] - pauli_ops[i])
            
            sum_term2 = SparsePauliOp("I" * num_qubits, 0)
            for p in range(max_runtime_diff):
                sum_term2 += (p + 1) * pauli_ops[num_machines * num_jobs + p + (j - 1) * max_runtime_diff]
            
            penalty_term += (sum_term1 + sum_term2) ** 2
        return penalty_term

    def _build_objective_term(self, num_jobs, num_machines, num_qubits, pauli_ops):
        """Builds the objective term to be minimized (e.g., makespan)."""
        objective_term = SparsePauliOp("I" * num_qubits, 0)
        for i in range(num_jobs):
            for j in range(num_machines):
                objective_term += pauli_ops[i + j * num_jobs]
        return objective_term
=======
    def binary_variable_operators(self, num_qubits):
        """
        Construct projector operators representing binary variables.

        For each qubit i, this returns an operator corresponding to
        (I - Z_i) / 2, realized as:

            0.5 * I - 0.5 * Z_i

        where Z_i is a Pauli-Z acting on qubit i and identity on all others.

        Args:
            num_qubits: Total number of qubits (binary variables).

        Returns:
            List of SparsePauliOp objects, one per qubit.
        """
        identity = SparsePauliOp.from_list([("I" * num_qubits, 0.5)])

        operator_list = []
        for i in range(num_qubits):
            pauli_str = ["I"] * num_qubits
            pauli_str[i] = "Z"
            label = "".join(pauli_str)
            z_term = SparsePauliOp.from_list([(label, -0.5)])
            operator_list.append(identity + z_term)
>>>>>>> Stashed changes

        return operator_list
