"""Support-vector identification from QP duals and geometric margins."""

from __future__ import annotations

import numpy as np

from module1.config.settings import SolverConfig
from module1.models.schemas import SupportVectorReport
from module1.optimization.formulation import QPData
from module1.optimization.solver import SolverResult


def analyze_support_vectors(
    result: SolverResult,
    qp: QPData,
    config: SolverConfig,
    max_indices: int = 200,
) -> tuple[SupportVectorReport, np.ndarray]:
    scores = qp.X @ result.w + result.bias
    margins = qp.y * scores
    notes: list[str] = []

    alpha = None
    if result.y_dual is not None and result.y_dual.shape[0] >= qp.n_variables + qp.n_samples:
        y_margin = result.y_dual[qp.n_variables : qp.n_variables + qp.n_samples]
        alpha = np.maximum(-y_margin, 0.0)
        tol = float(config.support_vector_alpha_tol) * float(np.mean(qp.C_samples))
        sv_mask = alpha > tol
        notes.append(
            "Support vectors are training rows with dual multiplier α_i above "
            f"{config.support_vector_alpha_tol} * mean(C_i)."
        )
    else:
        # Geometric proxy used only when duals are unavailable.
        sv_mask = margins <= 1.0 + 1e-3
        notes.append(
            "Dual multipliers were unavailable; support vectors are approximated as "
            "points with y(w^T x + b) <= 1 + 1e-3. This is a geometric proxy, not exact α."
        )

    indices = np.where(sv_mask)[0]
    count = int(indices.size)
    frac = count / qp.n_samples if qp.n_samples else 0.0
    mean_abs = float(np.mean(np.abs(margins[sv_mask]))) if count else None
    report = SupportVectorReport(
        count=count,
        fraction_of_training=float(frac),
        indices=indices[:max_indices].astype(int).tolist(),
        alpha_min=float(np.min(alpha[sv_mask])) if alpha is not None and count else None,
        alpha_max=float(np.max(alpha[sv_mask])) if alpha is not None and count else None,
        mean_abs_margin_on_sv=mean_abs,
        notes=notes,
    )
    return report, sv_mask
