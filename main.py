from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lsm_project_2.collocated.core.simple_algorithm import simple_algorithm
from lsm_project_2.collocated.linear_solvers import SOLVER_BACKEND
from lsm_project_2.collocated.mesh.mesh_loader import load_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal driver for the collocated SIMPLE solver."
    )
    parser.add_argument("--mesh", type=Path, required=True, help="Path to a Gmsh .msh file.")
    parser.add_argument(
        "--bc",
        type=Path,
        default=None,
        help="Boundary-condition YAML file (optional).",
    )
    parser.add_argument("--rho", type=float, default=1.0, help="Fluid density.")
    parser.add_argument(
        "--re", type=float, default=100.0, help="Reynolds number used to infer viscosity."
    )
    parser.add_argument(
        "--u-char",
        type=float,
        default=1.0,
        help="Characteristic velocity for viscosity scaling.",
    )
    parser.add_argument(
        "--l-char",
        type=float,
        default=1.0,
        help="Characteristic length for viscosity scaling.",
    )
    parser.add_argument(
        "--alpha-uv",
        type=float,
        default=0.7,
        help="Momentum under-relaxation factor.",
    )
    parser.add_argument(
        "--alpha-p",
        type=float,
        default=0.3,
        help="Pressure under-relaxation factor.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Maximum SIMPLE iterations.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-5,
        help="Relative residual tolerance for momentum equations.",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="Upwind",
        help="Convection discretisation scheme.",
    )
    parser.add_argument(
        "--limiter",
        type=str,
        default="MUSCL",
        help="Limiter name used for higher-order schemes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mesh = load_mesh(str(args.mesh), str(args.bc) if args.bc else None)

    mu = (args.u_char * args.l_char) / max(args.re, 1e-12)

    print(f"Loaded mesh with {mesh.cell_volumes.shape[0]} control volumes.")
    print(f"Using {SOLVER_BACKEND.upper()} backend for linear solves.")

    pressure, velocity, mdot, residuals, iterations, converged = simple_algorithm(
        mesh=mesh,
        rho=args.rho,
        mu=mu,
        alpha_uv=args.alpha_uv,
        alpha_p=args.alpha_p,
        max_iter=args.iterations,
        tol=args.tolerance,
        convection_scheme=args.scheme,
        limiter=args.limiter,
    )

    print(f"SIMPLE iterations: {iterations} ({'converged' if converged else 'stopped'})")
    print(
        f"Final residuals: u={residuals['u'][-1]:.2e}, "
        f"v={residuals['v'][-1]:.2e}, cont={residuals['continuity'][-1]:.2e}"
    )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / "p_final.npy", pressure)
    np.save(results_dir / "U_final.npy", velocity)
    np.save(results_dir / "mdot_final.npy", mdot)
    print("Saved fields to results/ directory.")


if __name__ == "__main__":
    main()
