"""KKT diagnostics for the split-variable soft-margin SVM QP.

OSQP solves:

    minimize  (1/2) z^T P z + q^T z
    subject to  l <= A z <= u

Its first-order stationarity condition is

    P z + q + A^T y  =  0

We report that residual directly (no fabricated numbers).  We also map the
last m duals of A to SVM multipliers α_i for the margin inequalities

    y_i (w^T x_i + b) + ξ_i >= 1

OSQP uses y such that Pz + q + A^T y = 0, with y ≤ 0 on constraints that only
have a finite lower bound.  Therefore α := max(-y_margin, 0) is the natural
non-negative multiplier.  Dual feasibility of the slack box requires
0 ≤ α_i ≤ C_i.

If duals are missing, we document that limitation and still report primal
feasibility of the original inequalities.
"""

from __future__ import annotations

import numpy as np

from module1.config.settings import SolverConfig
from module1.models.schemas import KKTReport
from module1.optimization.formulation import QPData
from module1.optimization.solver import SolverResult


def _finite_norm(vec: np.ndarray) -> float:
    if vec.size == 0:
        return 0.0
    return float(np.linalg.norm(vec, ord=np.inf))


def analyze_kkt(result: SolverResult, qp: QPData, config: SolverConfig) -> KKTReport:
    z = result.z
    w = result.w
    b = result.bias
    xi = result.xi
    X, y = qp.X, qp.y
    scores = X @ w + b
    margins = y * scores
    slack_residual = 1.0 - margins - xi  # should be <= 0
    max_margin_violation = float(np.max(np.maximum(slack_residual, 0.0)))
    max_xi_neg = float(np.max(np.maximum(-xi, 0.0)))
    max_wp_neg = float(np.max(np.maximum(-result.w_plus, 0.0)))
    max_wm_neg = float(np.max(np.maximum(-result.w_minus, 0.0)))
    max_nonneg = max(max_xi_neg, max_wp_neg, max_wm_neg)
    primal_feas = max(max_margin_violation, max_nonneg)

    limitations: list[str] = []
    stationarity = np.nan
    complementary = np.nan
    dual_feas = np.nan
    alpha_box = np.nan
    stationarity_b = np.nan

    if result.y_dual is None:
        limitations.append(
            "OSQP did not return dual variables; dual feasibility, complementary slackness, "
            "and stationarity of the QP Lagrangian cannot be computed exactly."
        )
        kkt_residual = primal_feas
    else:
        y_dual = result.y_dual
        if y_dual.shape[0] != qp.A.shape[0]:
            limitations.append(
                f"Dual vector length {y_dual.shape[0]} does not match A rows {qp.A.shape[0]}."
            )
        # P was stored as the upper triangle; restore the symmetric Hessian.
        P_full = qp.P + qp.P.T - sparse_diag_like(qp.P)
        residual = P_full @ z + qp.q + qp.A.T @ y_dual
        stationarity = _finite_norm(residual)

        n_var = qp.n_variables
        y_bounds = y_dual[:n_var]
        y_margin = y_dual[n_var : n_var + qp.n_samples]
        alpha = np.maximum(-y_margin, 0.0)
        alpha_box = float(
            np.max(np.maximum(-alpha, 0.0) + np.maximum(alpha - qp.C_samples, 0.0))
        )
        dual_feas = alpha_box
        stationarity_b = float(np.abs(np.sum(alpha * y)))

        Az = qp.A @ z
        # Complementary slackness for finite bounds: y_i (A_i z - bound) ≈ 0
        lower_gap = Az - qp.l
        upper_gap = qp.u - Az
        lower_gap = np.where(np.isfinite(qp.l), lower_gap, 0.0)
        upper_gap = np.where(np.isfinite(qp.u), upper_gap, 0.0)
        # OSQP stores one dual per row; pair its sign with the corresponding bound.
        cs = np.zeros_like(y_dual)
        lower_active = y_dual < 0
        upper_active = y_dual > 0
        cs[lower_active] = np.abs(y_dual[lower_active] * lower_gap[lower_active])
        cs[upper_active] = np.abs(y_dual[upper_active] * upper_gap[upper_active])
        complementary = float(np.max(cs)) if cs.size else 0.0

        kkt_residual = max(
            primal_feas,
            stationarity,
            complementary if np.isfinite(complementary) else 0.0,
            dual_feas if np.isfinite(dual_feas) else 0.0,
        )

        limitations.append(
            "Complementary slackness uses OSQP's bound duals. ADMM solutions may have "
            "moderate residuals even when status is 'solved'; polish reduces this."
        )
        limitations.append(
            "stationarity_b is |Σ α_i y_i|, the bias stationarity condition of the SVM."
        )

    tol = float(config.kkt_tolerance)
    kkt_residual = float(kkt_residual)
    within = bool(np.isfinite(kkt_residual) and kkt_residual <= tol)

    return KKTReport(
        primal_feasibility=float(primal_feas),
        dual_feasibility=float(dual_feas) if np.isfinite(dual_feas) else None,
        complementary_slackness=float(complementary) if np.isfinite(complementary) else None,
        stationarity_residual=float(stationarity) if np.isfinite(stationarity) else None,
        max_constraint_violation=float(max_margin_violation),
        kkt_residual=kkt_residual,
        within_tolerance=within,
        tolerance=tol,
        alpha_box_violation=float(alpha_box) if np.isfinite(alpha_box) else None,
        max_margin_violation=float(max_margin_violation),
        max_nonnegativity_violation=float(max_nonneg),
        stationarity_b=float(stationarity_b) if np.isfinite(stationarity_b) else None,
        limitations=limitations,
    )


def sparse_diag_like(P):
    """Extract the diagonal of a (possibly triangular) sparse P as a sparse diagonal."""
    from scipy import sparse

    diag = P.diagonal()
    return sparse.diags(diag, format="csc")
