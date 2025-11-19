"""
This module handles the generation of Hamiltonians for various quantum systems,
such as LiH and H4, using Qiskit and Tequila.
"""
import logging
from typing import Any, List, Tuple

import numpy as np
import tequila as tq
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem


class Hamiltonian_generation:
    """
    A class to generate molecular Hamiltonians for different bond distances.
    """

    def __init__(self, cfg: Any, logger: logging.Logger, main_folder: str):
        """
        Initializes the Hamiltonian_generation class.

        Args:
            cfg: A configuration object.
            logger: A logger for recording information.
            main_folder: The root directory for data files.
        """
        self.cfg = cfg
        self.logger = logger
        self.main_folder = main_folder

    def generate_Hamiltonian_LiH(self, bond_distance_range: np.ndarray, molecule_type: str) -> List[Tuple[np.ndarray, float, float]]:
        """
        Generates the qubit Hamiltonian for the LiH molecule at various bond distances.

        This method corrects for a phase change issue in PySCF for bond distances >= 2.5 Å
        by flipping the sign of the third orbital's integrals.

        Args:
            bond_distance_range: An array of bond distances to compute the Hamiltonian for.
            molecule_type: The specific type of LiH molecule ("LiH4" or "LiH6").

        Returns:
            A list of tuples, each containing the Hamiltonian matrix, ground state energy,
            and bond distance.
        """
        molecule_data = []
        PHASE_CHANGE_THRESHOLD = 2.5

        for r_bond in bond_distance_range:
            geom = f"Li 0.0 0.0 0.0\nH 0.0 0.0 {r_bond}"
            
            # Use Tequila to get one- and two-body integrals
            mol = tq.Molecule(
                geometry=geom,
                basis_set="sto-3g",
                active_orbitals=[1, 2, 5],
                transformation="JordanWigner"
            )
            
            nuclear_repulsion_energy = mol.get_enuc()
            one_body_integrals, two_body_integrals = mol.get_integrals()

            # Correct for the phase change in PySCF at large bond distances
            if r_bond >= PHASE_CHANGE_THRESHOLD:
                orbital_to_flip = 2
                one_body_integrals[orbital_to_flip, :] *= -1
                one_body_integrals[:, orbital_to_flip] *= -1
                two_body_integrals[orbital_to_flip, :, :, :] *= -1
                two_body_integrals[:, orbital_to_flip, :, :] *= -1
                two_body_integrals[:, :, orbital_to_flip, :] *= -1
                two_body_integrals[:, :, :, orbital_to_flip] *= -1

            # Construct the Fermionic Hamiltonian using Qiskit
            electronic_energy = ElectronicEnergy.from_raw_integrals(one_body_integrals, two_body_integrals)
            electronic_structure_problem = ElectronicStructureProblem(electronic_energy)
            fermionic_hamiltonian = electronic_structure_problem.hamiltonian.second_q_op()

            # Map the Fermionic Hamiltonian to a qubit Hamiltonian
            if molecule_type == "LiH6":
                mapper = JordanWignerMapper()
            elif molecule_type == "LiH4":
                mapper = ParityMapper(num_particles=(1, 1))
            else:
                raise ValueError(f"Unknown LiH molecule type: {molecule_type}")
                
            qubit_hamiltonian = mapper.map(fermionic_hamiltonian)
            
            # Add the nuclear repulsion energy
            identity_op = SparsePauliOp("I" * qubit_hamiltonian.num_qubits, nuclear_repulsion_energy)
            qubit_hamiltonian += identity_op
            
            hamiltonian_matrix = qubit_hamiltonian.to_matrix()
            
            # Calculate the ground state energy for reference
            eigenvalues, _ = np.linalg.eigh(hamiltonian_matrix)
            ground_state_energy = np.min(eigenvalues)

            molecule_data.append((hamiltonian_matrix, ground_state_energy, r_bond))
            self.logger.info(f"Generated Hamiltonian for LiH at R = {r_bond:.2f} Å, E0 = {ground_state_energy:.6f} Ha")

        return molecule_data

    def generate_Hamiltonian_H4(self, bond_distance_range: np.ndarray) -> List[Tuple[np.ndarray, float, float]]:
        """
        Generates the qubit Hamiltonian for the H4 molecule at various bond distances.

        Args:
            bond_distance_range: An array of bond distances to compute the Hamiltonian for.

        Returns:
            A list of tuples, each containing the Hamiltonian matrix, ground state energy,
            and bond distance.
        """
        molecule_data = []
        
        for r_bond in bond_distance_range:
            # Define the geometry of the H4 molecule (a square of side r_bond)
            geom = f"H 0.0 0.0 0.0\nH {r_bond} 0.0 0.0\nH 0.0 {r_bond} 0.0\nH {r_bond} {r_bond} 0.0"
            
            driver = PySCFDriver(atom=geom, basis="sto-3g", charge=0, spin=0)
            problem = driver.run()
            
            fermionic_hamiltonian = problem.hamiltonian.second_q_op()
            
            # Use ParityMapper to reduce the number of qubits
            mapper = ParityMapper(num_particles=(2, 2))
            qubit_hamiltonian = mapper.map(fermionic_hamiltonian)
            
            hamiltonian_matrix = qubit_hamiltonian.to_matrix()
            
            # Calculate the ground state energy
            eigenvalues, _ = np.linalg.eigh(hamiltonian_matrix)
            ground_state_energy = np.min(eigenvalues)

            molecule_data.append((hamiltonian_matrix, ground_state_energy, r_bond))
            self.logger.info(f"Generated Hamiltonian for H4 at R = {r_bond:.2f} Å, E0 = {ground_state_energy:.6f} Ha")
            
        return molecule_data