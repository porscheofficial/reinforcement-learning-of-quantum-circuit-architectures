# patch.py
# This file is used to monkey-patch the qiskit_algorithms library to fix an issue
# with the .H attribute on sparse matrices in recent versions of scipy.

from __future__ import annotations
import numpy as np
from scipy import sparse as scisparse

from qiskit_algorithms.eigensolvers.numpy_eigensolver import NumPyEigensolver

def _solve_sparse_patched(op_matrix: scisparse.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Patched version of _solve_sparse to use .conj().T instead of .H"""
    if (op_matrix != op_matrix.conj().T).nnz == 0:
        # Operator is Hermitian
        return scisparse.linalg.eigsh(op_matrix, k=k, which="SA")
    else:
        return scisparse.linalg.eigs(op_matrix, k=k, which="SR")

def _solve_dense_patched(op_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Patched version of _solve_dense to use np.allclose for hermitian check."""
    if np.allclose(op_matrix, op_matrix.conj().T):
        # Operator is Hermitian
        return np.linalg.eigh(op_matrix)
    else:
        return np.linalg.eig(op_matrix)

# Apply the patches as static methods
NumPyEigensolver._solve_sparse = staticmethod(_solve_sparse_patched)
NumPyEigensolver._solve_dense = staticmethod(_solve_dense_patched)
