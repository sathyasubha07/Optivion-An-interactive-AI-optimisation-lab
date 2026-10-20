"""OSQP wrapper for the SVM+L1 quadratic program."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from module1.config.settings import SolverConfig
from module1.exceptions import OptimizationError
from module1.logging_setup import get_logger, log_kv
from module1.optimization.formulation import QPData, primal_objective, unpack_primal

logger = get_logger(__name__)


@dataclass
class SolverResult:
    status: str
    status_val: int | None
    converged: bool
    z: np.ndarray
    y_dual: np.ndarray | None
    w_plus: np.ndarray
    w_minus: np.ndarray
    w: np.ndarray
    bias: float
    xi: np.ndarray
    objective_value: float | None
    primal_objective: float
    iterations: int | None
    training_time_seconds: float
    polish_status: str | None
    primal_residual: float | None
    dual_residual: float | None
    run_time: float | None
    notes: list[str] = field(default_factory=list)
    qp: QPData | None = None


def _osqp_setup_and_solve(qp: QPData, config: SolverConfig):
    try:
        import osqp
    except ImportError as exc:
        raise OptimizationError(
            "The OSQP package is required to solve the SVM+L1 QP. Install with: pip install osqp"
        ) from exc

    modern = dict(
        max_iter=int(config.max_iter),
        eps_abs=float(config.eps_abs),
        eps_rel=float(config.eps_rel),
        polishing=bool(config.polish),
        verbose=bool(config.verbose),
        warm_starting=False,
    )
    legacy = dict(
        max_iter=int(config.max_iter),
        eps_abs=float(config.eps_abs),
        eps_rel=float(config.eps_rel),
        polish=bool(config.polish),
        verbose=bool(config.verbose),
        warm_start=False,
    )

    if not hasattr(osqp, "OSQP"):
        raise OptimizationError("Unsupported OSQP installation: missing OSQP class.")

    solver = osqp.OSQP()
    last_error: Exception | None = None
    for settings in (modern, legacy):
        try:
            solver = osqp.OSQP()
            try:
                solver.setup(qp.P, qp.q, qp.A, qp.l, qp.u, **settings)
            except TypeError:
                solver.setup(P=qp.P, q=qp.q, A=qp.A, l=qp.l, u=qp.u, **settings)
            try:
                return solver.solve(raise_error=False)
            except TypeError:
                return solver.solve()
        except TypeError as exc:
            last_error = exc
            continue
    raise OptimizationError(f"OSQP setup failed: {last_error}")


def solve_qp(qp: QPData, config: SolverConfig) -> SolverResult:
    log_kv(
        logger,
        "solver_start",
        n_samples=qp.n_samples,
        n_features=qp.n_features,
        n_variables=qp.n_variables,
        C=qp.C,
        lambda_l1=qp.lambda_l1,
    )
    t0 = time.perf_counter()
    try:
        raw = _osqp_setup_and_solve(qp, config)
    except OptimizationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise OptimizationError(f"OSQP failed: {exc}") from exc
    elapsed = time.perf_counter() - t0

    info = getattr(raw, "info", None)
    status = str(getattr(info, "status", getattr(raw, "status", "unknown")))
    status_val = getattr(info, "status_val", None)
    z = np.asarray(raw.x, dtype=np.float64) if raw.x is not None else None
    y_dual = np.asarray(raw.y, dtype=np.float64) if getattr(raw, "y", None) is not None else None

    acceptable = {
        "solved",
        "solved inaccurate",
        "optimal",
        "optimal inaccurate",
        "solved (inaccurate)",
    }
    status_l = status.lower()
    if z is None or not np.isfinite(z).all():
        raise OptimizationError(f"Solver returned a non-finite primal solution (status={status}).")
    if status_l not in acceptable and "solved" not in status_l and "optimal" not in status_l:
        raise OptimizationError(
            f"OSQP did not converge (status={status}). "
            "Check C, lambda, scaling, and sample size. The error is not hidden."
        )

    w_plus, w_minus, bias, xi = unpack_primal(z, qp.n_features, qp.n_samples)
    xi = np.maximum(xi, 0.0)
    w = w_plus - w_minus
    pobj = primal_objective(w, xi, qp.C_samples, qp.lambda_l1)
    obj = getattr(info, "obj_val", None)
    notes = [
        "Solved the split-variable QP with OSQP (ADMM). "
        "This is the L1-reformulated soft-margin SVM, not sklearn.svm.SVC."
    ]
    if qp.lambda_l1 == 0.0:
        notes.append(
            "lambda_l1=0: the split (w⁺, w⁻) is not unique. Weights are reported as w = w⁺ − w⁻."
        )
    log_kv(logger, "solver_status", status=status, objective=pobj, seconds=round(elapsed, 4))
    return SolverResult(
        status=status,
        status_val=int(status_val) if status_val is not None else None,
        converged=True,
        z=z,
        y_dual=y_dual,
        w_plus=w_plus,
        w_minus=w_minus,
        w=w,
        bias=bias,
        xi=xi,
        objective_value=float(obj) if obj is not None and np.isfinite(obj) else None,
        primal_objective=float(pobj),
        iterations=int(getattr(info, "iter", 0) or 0),
        training_time_seconds=float(elapsed),
        polish_status=str(getattr(info, "status_polish", None)),
        primal_residual=(
            float(getattr(info, "prim_res", getattr(info, "pri_res", np.nan)))
            if info is not None and np.isfinite(getattr(info, "prim_res", getattr(info, "pri_res", np.nan)))
            else None
        ),
        dual_residual=(
            float(getattr(info, "dual_res", getattr(info, "dua_res", np.nan)))
            if info is not None and np.isfinite(getattr(info, "dual_res", getattr(info, "dua_res", np.nan)))
            else None
        ),
        run_time=(
            float(getattr(info, "run_time", elapsed))
            if info is not None and np.isfinite(getattr(info, "run_time", elapsed))
            else float(elapsed)
        ),
        notes=notes,
        qp=qp,
    )
