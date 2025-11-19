"""
This script monkey-patches the `qiskit_algorithms.eigensolvers.NumPyEigensolver`
to address an incompatibility with recent versions of `scipy`.

The `.H` attribute for the conjugate transpose of sparse matrices was deprecated in
`scipy` and removed in later versions, causing errors in `qiskit-algorithms`.
This patch replaces the use of `.H` with `.conj().T` for sparse matrices and
improves the check for Hermitian matrices for dense arrays.
"""

from __future__ import annotations
import numpy as np
from scipy import sparse as scisparse

from qiskit_algorithms.eigensolvers.numpy_eigensolver import NumPyEigensolver

def _solve_sparse_patched(op_matrix: scisparse.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    A patched version of `_solve_sparse` that uses `.conj().T` instead of `.H`.

    This function checks if the operator is Hermitian and uses the appropriate
    scipy eigensolver (`eigsh` for Hermitian, `eigs` for non-Hermitian).

    Args:
        op_matrix: The sparse matrix representation of the operator.
        k: The number of eigenvalues to compute.

    Returns:
        A tuple containing the computed eigenvalues and eigenvectors.
    """
    # Check if the operator is Hermitian by comparing it to its conjugate transpose.
    if (op_matrix != op_matrix.conj().T).nnz == 0:
        # Operator is Hermitian, use the more efficient eigsh solver.
        return scisparse.linalg.eigsh(op_matrix, k=k, which="SA")
    else:
        # Operator is not Hermitian, use the general eigs solver.
        return scisparse.linalg.eigs(op_matrix, k=k, which="SR")

def _solve_dense_patched(op_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    A patched version of `_solve_dense` that uses `np.allclose` for the Hermitian check.

    This provides a more robust way to check for Hermitian matrices with floating-point
    numbers.

    Args:
        op_matrix: The dense matrix representation of the operator.

    Returns:
        A tuple containing the computed eigenvalues and eigenvectors.
    """
    # Check if the operator is Hermitian using a tolerance for floating-point comparisons.
    if np.allclose(op_matrix, op_matrix.conj().T):
        # Operator is Hermitian, use the more efficient eigh solver.
        return np.linalg.eigh(op_matrix)
    else:
        # Operator is not Hermitian, use the general eig solver.
        return np.linalg.eig(op_matrix)

# --- Apply the Patches ---
# Replace the original methods in NumPyEigensolver with the patched versions.
# These methods are static, so we wrap our functions with staticmethod().
NumPyEigensolver._solve_sparse = staticmethod(_solve_sparse_patched)
NumPyEigensolver._solve_dense = staticmethod(_solve_dense_patched)

print("Applied patch to qiskit_algorithms.NumPyEigensolver.")
