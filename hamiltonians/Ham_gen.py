from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from numpy import linalg as LA
import matplotlib.pyplot as plt
import numpy as np
import tequila as tq
import os


class Hamiltonian_generation:
    """
    Generation of problem Hamiltonians and consistency checks.

    This class provides:
      - LiH Hamiltonian generation with a phase-consistency correction
        for one orbital at large bond distances.
      - H4-based Hamiltonians with and without optimized orbitals.
      - Optional diagnostic checks for the LiH Hamiltonian modification.
    """

    def __init__(self, cfg, logger, main_folder):
        """
        Args:
            cfg: Configuration object/module with 'characteristics' and related keys.
            logger: Logger instance used for diagnostic output.
            main_folder: Directory where diagnostic plots are stored.
        """
        self.cfg = cfg
        self.logger = logger
        self.main_folder = main_folder

    # ---------------------------------------------------------------------
    # LiH Hamiltonian generation
    # ---------------------------------------------------------------------
    def generate_Hamiltonian_LiH(self, bond_distance_range, molecule):
        """
        Generate LiH Hamiltonians on a given bond-distance grid.

        For bond distances above a given threshold, a phase change appears
        in one orbital when using PySCF. This is corrected by flipping the
        sign of one orbital in the one- and two-body integrals to ensure a
        smooth potential energy curve.

        Args:
            bond_distance_range: 1D iterable of bond distances.
            molecule: String identifier ('LiH4', 'LiH6') used only
                      to steer mapping conditions (kept for compatibility).

        Returns:
            List of tuples (matrix, total_energy, bond_distance),
            where:
              - matrix is the qubit Hamiltonian as a dense numpy array,
              - total_energy is the ground-state energy (including constant),
              - bond_distance is rounded to two decimals.
        """
        ham = []
        ham_modified = []
        molecule_data = []
        threshold = 2.5

        """
        When generating the Hamiltonian for LiH using PySCF, a phase change
        occurs in the third orbital for bond distances ≥ 2.5. This results in
        an abrupt sign change in the Hamiltonian coefficients. To ensure a
        consistent energy surface across bond distances, this issue is
        corrected by flipping the sign of the third orbital in the one- and
        two-body integrals for bond distances > 2.5.
        """

        # Determine the grid point at which the modification becomes active
        if bond_distance_range[-1] < threshold or bond_distance_range[0] < threshold:
            coeff_threshold = threshold
        else:
            target = threshold
            coeff_threshold = np.round(
                bond_distance_range[bond_distance_range >= target][0], 3
            )

        for R in bond_distance_range:
            geom = "Li 0.0 0.0 0.0\nH 0.0 0.0 {}".format(R)

            # Tequila is used to generate the one- and two-body integrals,
            # which are independent of the chosen qubit encoding.
            encoding = "JordanWigner"  # kept for consistency; not used directly
            mol = tq.Molecule(
                geometry=geom.format(R=R),
                basis_set="sto-3g",
                active_orbitals=[1, 2, 5],
                backend="pyscf",
                transformation="JordanWigner",
            )
            H = mol.make_hamiltonian()

            # Ordering used in Qiskit (Mulliken)
            ordering = "mulliken"
            constant, one_body, two_body = mol.get_integrals(
                two_body_ordering=ordering
            )

            # Extract raw two-body elements from Tequila container
            two_body = two_body.elems

            # Original (unmodified) Hamiltonian via Qiskit
            electronic_energy = ElectronicEnergy.from_raw_integrals(
                one_body, two_body
            )
            electronic_structure_problem = ElectronicStructureProblem(
                electronic_energy
            )
            fermionic_hamiltonian = (
                electronic_structure_problem.hamiltonian.second_q_op()
            )

            # Choice of mapper depends on the system size
            if self.cfg.characteristics["system"] == "LiH6":
                mapper = JordanWignerMapper()
            elif self.cfg.characteristics["system"] == "LiH4":
                mapper = ParityMapper(num_particles=(1, 1))
            else:
                mapper = ParityMapper(num_particles=(1, 1))

            qubit_hamiltonian = mapper.map(fermionic_hamiltonian)
            ham.append(qubit_hamiltonian)

            # Add nuclear repulsion constant
            identity_op = SparsePauliOp.from_list(
                [("I" * qubit_hamiltonian.num_qubits, constant)]
            )
            qubit_hamiltonian += identity_op

            # -----------------------------------------------------------------
            # Modification: change sign of the third orbital above threshold
            # -----------------------------------------------------------------
            if np.round(R, 3) == coeff_threshold or np.round(R, 3) > coeff_threshold:
                orbital_index = 2

                # One-body integrals
                one_body[orbital_index, :] *= -1
                one_body[:, orbital_index] *= -1

                # Two-body integrals
                two_body[orbital_index, :, :, :] *= -1
                two_body[:, orbital_index, :, :] *= -1
                two_body[:, :, orbital_index, :] *= -1
                two_body[:, :, :, orbital_index] *= -1

            # Modified Hamiltonian
            electronic_energy = ElectronicEnergy.from_raw_integrals(
                one_body, two_body
            )
            electronic_structure_problem = ElectronicStructureProblem(
                electronic_energy
            )
            fermionic_hamiltonian = (
                electronic_structure_problem.hamiltonian.second_q_op()
            )

            if self.cfg.characteristics["system"] == "LiH6":
                mapper = JordanWignerMapper()
            elif self.cfg.characteristics["system"] == "LiH4":
                mapper = ParityMapper(num_particles=(1, 1))
            else:
                mapper = ParityMapper(num_particles=(1, 1))

            qubit_hamiltonian = mapper.map(fermionic_hamiltonian)
            ham_modified.append(qubit_hamiltonian)

            identity_op = SparsePauliOp.from_list(
                [("I" * qubit_hamiltonian.num_qubits, constant)]
            )
            qubit_hamiltonian += identity_op

            matrix = qubit_hamiltonian.to_matrix()
            solver = GroundStateEigensolver(mapper, NumPyMinimumEigensolver())
            result = solver.solve(electronic_structure_problem)

            molecule_data.append(
                (matrix, result.total_energies[0] + constant, np.round(R, 2))
            )

        if self.cfg.characteristics["ham_check"] == "on":
            self.check_modification(ham, ham_modified, bond_distance_range)

        return molecule_data

    # ---------------------------------------------------------------------
    # H4 / H48 Hamiltonian generation
    # ---------------------------------------------------------------------
    def generate_Hamiltonian_H48(self, bond_distance_range):
        """
        Generate Hamiltonians for the H4 chain at equally spaced positions.

        Depending on cfg.characteristics["system"], either:
          - 'H48': SPA-based optimized orbitals,
          - 'H48HF': canonical HF/STO-3G orbitals.

        Args:
            bond_distance_range: 1D iterable of bond distances.

        Returns:
            List of tuples (H_matrix, energy, bond_distance),
            where H_matrix is the dense Hamiltonian matrix in the chosen
            orbital basis.
        """
        terms = {}
        molecule_data = []

        if self.cfg.characteristics["system"] == "H48":
            for R in bond_distance_range:
                geometry = (
                    f"h 0.0 0.0 0.0\n"
                    f"h 0.0 0.0 {np.round(R, 2)}\n"
                    f"h 0.0 0.0 {np.round(2 * R, 2)}\n"
                    f"h 0.0 0.0 {np.round(3 * R, 2)}"
                )
                mol = tq.Molecule(geometry=geometry, basis_set="sto-3g")
                energy = mol.compute_energy("fci")

                # Switch from canonical HF orbitals to orthonormalized STO-3G
                # orbitals to follow the notation in the reference article.
                mol = mol.use_native_orbitals()
                USPA = mol.make_ansatz(name="SPA", edges=[(0, 1), (2, 3)])

                guess = np.eye(4)
                guess[0] = [1.0, 1.0, 0.0, 0.0]
                guess[1] = [1.0, -1.0, 0.0, 0.0]
                guess[2] = [0.0, 0.0, 1.0, 1.0]
                guess[3] = [0.0, 0.0, 1.0, -1.0]

                opt = tq.chemistry.optimize_orbitals(
                    mol, circuit=USPA, initial_guess=guess, silent=True
                )
                H = opt.molecule.make_hamiltonian()
                HH = H.to_matrix()

                molecule_data.append((HH, energy, np.round(R, 2)))

        elif self.cfg.characteristics["system"] == "H48HF":
            for R in bond_distance_range:
                geometry = (
                    f"h 0.0 0.0 0.0\n"
                    f"h 0.0 0.0 {np.round(R, 2)}\n"
                    f"h 0.0 0.0 {np.round(2 * R, 2)}\n"
                    f"h 0.0 0.0 {np.round(3 * R, 2)}"
                )
                mol = tq.Molecule(geometry=geometry, basis_set="sto-3g")
                energy = mol.compute_energy("fci")
                H = mol.make_hamiltonian()
                HH = H.to_matrix()

                molecule_data.append((HH, energy, np.round(R, 2)))

        return molecule_data

    # ---------------------------------------------------------------------
    # Diagnostic check for LiH Hamiltonian modification
    # ---------------------------------------------------------------------
    def check_modification(self, ham, ham_modified, bond_distance_range):
        """
        Compare original and modified LiH Hamiltonians across bond distances.

        This method:
          - logs and plots coefficients of Pauli terms as functions of R,
          - checks whether the modified Hamiltonians are unitarily equivalent
            to the originals and have matching spectra.
        """
        terms = {}
        terms_modified = {}
        self.logger.info("Testing Hamiltonian modifications...")

        # Coefficients of original Hamiltonians
        for i in range(len(ham)):
            for label, coeff in zip(ham[i].paulis.to_labels(), ham[i].coeffs):
                if label in terms:
                    terms[label] += [np.real(coeff)]
                else:
                    terms[label] = [np.real(coeff)]

        # Coefficients of modified Hamiltonians
        for i in range(len(ham_modified)):
            for label, coeff in zip(
                ham_modified[i].paulis.to_labels(), ham_modified[i].coeffs
            ):
                if label in terms_modified:
                    terms_modified[label] += [np.real(coeff)]
                else:
                    terms_modified[label] = [np.real(coeff)]

        # Plot coefficient evolution before and after modification
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for k, v in terms.items():
            axes[0].plot(bond_distance_range, v, label=k, marker="x")
        axes[0].set_xlabel("Bond distance")
        axes[0].set_ylabel("Coefficients")
        axes[0].set_title("Original coefficients")
        axes[0].set_ylim(-0.1, 0.1)

        for k, v in terms_modified.items():
            axes[1].plot(bond_distance_range, v, label=k, marker="x")
        axes[1].set_xlabel("Bond distance")
        axes[1].set_ylabel("Coefficients")
        axes[1].set_title("Modified coefficients")
        axes[1].set_ylim(-0.1, 0.1)

        plt.tight_layout()
        path = os.path.join(self.main_folder, "ham_coeffs_plot.png")
        plt.savefig(path)

        # Check matrix equality and unitary equivalence
        for Aq, Bq, val in zip(ham, ham_modified, bond_distance_range):
            A = Aq.to_matrix()
            B = Bq.to_matrix()

            self.logger.info(f"Bond distance: {np.round(val, 3)}")

            if not np.allclose(A, B, rtol=1e-4, atol=1e-6):
                self.logger.info("Hamiltonians are not the same.")
            else:
                self.logger.info("Hamiltonians are the same.")

            eigvals_A, eigvecs_A = LA.eigh(A)
            eigvals_B, eigvecs_B = LA.eigh(B)

            if np.allclose(
                np.sort(eigvals_A),
                np.sort(eigvals_B),
                rtol=1e-4,
                atol=1e-4,
            ):
                self.logger.info("Spectrum matches!")

            U = eigvecs_B @ eigvecs_A.T.conj()

            if (
                np.allclose(
                    A,
                    U.T.conj() @ B @ U,
                    rtol=1e-4,
                    atol=1e-6,
                )
                and np.allclose(
                    U @ U.T.conj(),
                    np.eye(np.shape(A)[0]),
                    rtol=1e-4,
                    atol=1e-6,
                )
                and np.allclose(
                    U.T.conj() @ U,
                    np.eye(np.shape(A)[0]),
                    rtol=1e-4,
                    atol=1e-6,
                )
            ):
                self.logger.info("Unitary equivalence!")

        return None
