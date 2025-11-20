from qiskit.quantum_info import SparsePauliOp
import numpy as np
from numpy import linalg as LA


class JSP_generation:
    """
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
        """
        self.cfg = cfg
        self.logger = logger
        self.main_folder = main_folder

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

        return JSP_info

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

        return operator_list
