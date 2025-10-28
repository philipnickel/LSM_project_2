from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def scipy_solver(
    A_csr: csr_matrix,
    b_np: np.ndarray,
    tolerance: float = 1.0e-10,
    max_iterations: int = 0,
    **_,
):
    """
    Simple wrapper around SciPy's sparse direct solver.

    Parameters
    ----------
    A_csr : csr_matrix
        System matrix.
    b_np : ndarray
        Right-hand side.
    tolerance, max_iterations :
        Accepted for API compatibility but unused.

    Returns
    -------
    x_np : ndarray
        Solution vector.
    residual_norm : float
        Euclidean norm of the residual.
    context : None
        Placeholder to mirror the PETSc solver signature.
    """
    x_np = spsolve(A_csr, b_np)
    residual = b_np - A_csr @ x_np
    return x_np, float(np.linalg.norm(residual)), None
