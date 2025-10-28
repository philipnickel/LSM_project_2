from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from .scipy_solver import scipy_solver

try:  # pragma: no cover - optional dependency
    from .petsc_solver import petsc_solver as _petsc_solver

    HAS_PETSC = True
except Exception:  # pylint: disable=broad-except
    _petsc_solver = None
    HAS_PETSC = False

SOLVER_BACKEND = "petsc" if HAS_PETSC else "scipy"


def solve_linear_system(
    A_csr: csr_matrix,
    b_np: np.ndarray,
    *,
    use_petsc: bool | None = None,
    **solver_kwargs: Any,
) -> Tuple[np.ndarray, float, Any]:
    """
    Solve A x = b using the best available backend.

    Parameters
    ----------
    A_csr, b_np :
        Linear system in CSR / dense format.
    use_petsc : bool or None
        Force or disable PETSc. Defaults to ``True`` when PETSc is available.
    solver_kwargs :
        Forwarded to the backend solver.
    """
    if use_petsc is None:
        use_petsc = HAS_PETSC

    if use_petsc and HAS_PETSC:
        return _petsc_solver(A_csr, b_np, **solver_kwargs)  # type: ignore[misc]

    # Drop PETSc-only hints that SciPy cannot use.
    solver_kwargs.pop("solver_type", None)
    solver_kwargs.pop("preconditioner", None)
    solver_kwargs.pop("remove_nullspace", None)
    return scipy_solver(A_csr, b_np, **solver_kwargs)


__all__ = ["solve_linear_system", "SOLVER_BACKEND", "HAS_PETSC"]
