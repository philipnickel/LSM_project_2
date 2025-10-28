from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from lsm_project_2.collocated.assembly.convection_diffusion_matrix import (
    assemble_diffusion_convection_matrix,
)
from lsm_project_2.collocated.assembly.divergence import (
    compute_divergence_from_face_fluxes,
)
from lsm_project_2.collocated.assembly.pressure_correction_eq_assembly import (
    assemble_pressure_correction_matrix,
)
from lsm_project_2.collocated.assembly.rhie_chow import (
    mdot_calculation,
    rhie_chow_velocity,
)
from lsm_project_2.collocated.core.corrections import velocity_correction
from lsm_project_2.collocated.core.helpers import (
    bold_Dv_calculation,
    compute_l2_norm,
    compute_residual,
    get_unique_cells_from_faces,
    interpolate_to_face,
    relax_momentum_equation,
)
from lsm_project_2.collocated.discretization.gradient.leastSquares import (
    compute_cell_gradients,
)
from lsm_project_2.collocated.linear_solvers import solve_linear_system


@dataclass
class SolverState:
    """Holds the primary fields for the SIMPLE loop."""

    pressure: np.ndarray
    velocity: np.ndarray
    face_velocity: np.ndarray
    mass_flux: np.ndarray


def _assemble_pressure_matrix(mesh, rho: float, reference_cell: int) -> csr_matrix:
    """Build the pressure correction matrix with a fixed reference cell."""
    n_cells = mesh.cell_volumes.shape[0]
    row, col, data = assemble_pressure_correction_matrix(mesh, rho)
    A_p = coo_matrix((data, (row, col)), shape=(n_cells, n_cells)).tocsr()

    # Enforce p' = 0 in the reference cell to remove nullspace
    A_p = A_p.tolil()
    A_p[reference_cell, :] = 0.0
    A_p[:, reference_cell] = 0.0
    A_p[reference_cell, reference_cell] = 1.0
    return A_p.tocsr()


def _relative_l2(residual: float, rhs_norm: float) -> float:
    denom = rhs_norm if rhs_norm > 1e-30 else 1.0
    return residual / denom


def simple_algorithm(
    mesh,
    rho: float,
    mu: float,
    alpha_uv: float = 0.7,
    alpha_p: float = 0.3,
    max_iter: int = 500,
    tol: float = 1.0e-6,
    convection_scheme: str = "Upwind",
    limiter: str = "MUSCL",
    linear_solver_settings: Optional[Dict[str, Dict[str, float]]] = None,
    reference_cell: int = 0,
    progress_callback: Optional[Callable[[int, float, float, float], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], int, bool]:
    """
    Steady SIMPLE algorithm for the collocated arrangement.

    Parameters
    ----------
    mesh : MeshData2D
        Geometry and connectivity description of the domain.
    rho, mu : float
        Density and viscosity.
    alpha_uv, alpha_p : float
        Under-relaxation factors for momentum and pressure.
    max_iter : int
        Maximum SIMPLE iterations.
    tol : float
        Target relative L2 norm for u- and v-momentum residuals.
    convection_scheme, limiter : str
        Options forwarded to the convection assembler.
    linear_solver_settings : dict, optional
        Per-equation solver keyword arguments.
    reference_cell : int
        Cell index pinned to zero pressure correction.
    progress_callback : callable, optional
        Receives (iteration, u_res, v_res, cont_res).

    Returns
    -------
    p : ndarray
        Final pressure field.
    U : ndarray
        Final velocity field (n_cells, 2).
    mdot : ndarray
        Mass flux at faces.
    residuals : dict
        History arrays keyed by 'u', 'v', 'continuity'.
    iterations : int
        Number of completed SIMPLE iterations.
    converged : bool
        Whether the tolerance criterion was met.
    """
    if linear_solver_settings is None:
        linear_solver_settings = {
            "momentum": {},
            "pressure": {},
        }

    n_cells = mesh.cell_volumes.shape[0]
    n_faces = mesh.face_centers.shape[0]
    internal_cells = get_unique_cells_from_faces(mesh, mesh.internal_faces)

    state = SolverState(
        pressure=np.zeros(n_cells),
        velocity=np.zeros((n_cells, 2)),
        face_velocity=np.zeros((n_faces, 2)),
        mass_flux=np.zeros(n_faces),
    )

    pressure_matrix = _assemble_pressure_matrix(mesh, rho, reference_cell)

    u_history = np.zeros(max_iter)
    v_history = np.zeros(max_iter)
    cont_history = np.zeros(max_iter)

    converged = False

    for iteration in range(max_iter):
        # Momentum prediction -------------------------------------------------
        grad_p = compute_cell_gradients(
            mesh, state.pressure, weighted=True, weight_exponent=0.5, use_limiter=False
        )
        grad_u = compute_cell_gradients(
            mesh, state.velocity[:, 0], weighted=True, weight_exponent=0.5, use_limiter=True
        )
        grad_v = compute_cell_gradients(
            mesh, state.velocity[:, 1], weighted=True, weight_exponent=0.5, use_limiter=True
        )

        # U-momentum
        row_u, col_u, data_u, b_u = assemble_diffusion_convection_matrix(
            mesh,
            state.mass_flux,
            grad_u,
            state.velocity,
            rho,
            mu,
            component_idx=0,
            phi=state.velocity[:, 0],
            scheme=convection_scheme,
            limiter=limiter,
            pressure_field=state.pressure,
            grad_pressure_field=grad_p,
            transient=False,
        )
        A_u = coo_matrix((data_u, (row_u, col_u)), shape=(n_cells, n_cells)).tocsr()
        A_u_diag = A_u.diagonal()
        rhs_u = b_u - grad_p[:, 0] * mesh.cell_volumes
        rhs_u_unrelaxed = rhs_u.copy()
        relaxed_diag_u, rhs_u = relax_momentum_equation(
            rhs_u, A_u_diag, state.velocity[:, 0], alpha_uv
        )
        A_u.setdiag(relaxed_diag_u)
        U_star_x, _, _ = solve_linear_system(
            A_u, rhs_u, **linear_solver_settings.get("momentum", {})
        )
        A_u.setdiag(A_u_diag)

        # Residual (relative L2)
        u_residual, _ = compute_residual(
            A_u.data, A_u.indices, A_u.indptr, U_star_x, rhs_u_unrelaxed, norm_indices=internal_cells
        )
        u_rhs_norm = compute_l2_norm(rhs_u_unrelaxed, internal_cells)
        u_history[iteration] = _relative_l2(u_residual, u_rhs_norm)

        # V-momentum
        row_v, col_v, data_v, b_v = assemble_diffusion_convection_matrix(
            mesh,
            state.mass_flux,
            grad_v,
            state.velocity,
            rho,
            mu,
            component_idx=1,
            phi=state.velocity[:, 1],
            scheme=convection_scheme,
            limiter=limiter,
            pressure_field=state.pressure,
            grad_pressure_field=grad_p,
            transient=False,
        )
        A_v = coo_matrix((data_v, (row_v, col_v)), shape=(n_cells, n_cells)).tocsr()
        A_v_diag = A_v.diagonal()
        rhs_v = b_v - grad_p[:, 1] * mesh.cell_volumes
        rhs_v_unrelaxed = rhs_v.copy()
        relaxed_diag_v, rhs_v = relax_momentum_equation(
            rhs_v, A_v_diag, state.velocity[:, 1], alpha_uv
        )
        A_v.setdiag(relaxed_diag_v)
        U_star_y, _, _ = solve_linear_system(
            A_v, rhs_v, **linear_solver_settings.get("momentum", {})
        )
        A_v.setdiag(A_v_diag)

        v_residual, _ = compute_residual(
            A_v.data, A_v.indices, A_v.indptr, U_star_y, rhs_v_unrelaxed, norm_indices=internal_cells
        )
        v_rhs_norm = compute_l2_norm(rhs_v_unrelaxed, internal_cells)
        v_history[iteration] = _relative_l2(v_residual, v_rhs_norm)

        U_star = np.column_stack((U_star_x, U_star_y))

        # Rhie–Chow interpolation ---------------------------------------------
        bold_D = bold_Dv_calculation(mesh, A_u_diag, A_v_diag)
        bold_D_face = interpolate_to_face(mesh, bold_D)
        grad_p_face = interpolate_to_face(mesh, grad_p)
        U_star_bar = interpolate_to_face(mesh, U_star)
        U_old_bar = interpolate_to_face(mesh, state.velocity)

        U_faces_star = rhie_chow_velocity(
            mesh,
            U_star,
            U_star_bar,
            U_old_bar,
            state.face_velocity,
            grad_p_face,
            grad_p,
            state.pressure,
            alpha_uv,
            bold_D_face,
        )
        mdot_star = mdot_calculation(mesh, rho, U_faces_star, correction=False)

        # Pressure correction -------------------------------------------------
        rhs_p = compute_divergence_from_face_fluxes(mesh, mdot_star)
        cont_history[iteration] = compute_l2_norm(rhs_p, internal_cells)

        rhs_p[reference_cell] = 0.0
        p_prime, _, _ = solve_linear_system(
            pressure_matrix, rhs_p, **linear_solver_settings.get("pressure", {})
        )
        p_prime[reference_cell] = 0.0

        grad_p_prime = compute_cell_gradients(
            mesh, p_prime, weighted=True, weight_exponent=0.5, use_limiter=False
        )
        U_prime = velocity_correction(mesh, grad_p_prime, bold_D)
        U_prime_face = interpolate_to_face(mesh, U_prime)

        # Field updates -------------------------------------------------------
        state.velocity = U_star + U_prime
        state.face_velocity = U_faces_star + U_prime_face
        state.mass_flux = mdot_star + mdot_calculation(
            mesh, rho, U_prime_face, correction=True
        )
        state.pressure = state.pressure + alpha_p * p_prime

        if progress_callback is not None:
            progress_callback(iteration, u_history[iteration], v_history[iteration], cont_history[iteration])

        if max(u_history[iteration], v_history[iteration]) < tol:
            converged = True
            final_iteration = iteration + 1
            break

    else:
        final_iteration = max_iter

    residuals = {
        "u": u_history[:final_iteration],
        "v": v_history[:final_iteration],
        "continuity": cont_history[:final_iteration],
    }

    return (
        state.pressure,
        state.velocity,
        state.mass_flux,
        residuals,
        final_iteration,
        converged,
    )
