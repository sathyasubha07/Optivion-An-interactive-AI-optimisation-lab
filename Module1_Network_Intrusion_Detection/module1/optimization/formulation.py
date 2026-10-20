"""Quadratic-program formulation of a soft-margin linear SVM with L1 sparsity.

Intended problem (non-smooth L1):

    minimize_{w, b, ξ}
        (1/2) ||w||_2^2  +  C * Σ_i ξ_i  +  λ ||w||_1
    subject to
        y_i (w^T x_i + b) >= 1 - ξ_i
        ξ_i >= 0

The L1 term is not differentiable at 0, so it must NOT be handed to a smooth
QP solver as "λ Σ |w_j|".  We use the standard splitting:

    w = w⁺ − w⁻,    w⁺ >= 0,    w⁻ >= 0
    ||w||_1 = 1^T (w⁺ + w⁻)

because at optimality at most one of w⁺_j, w⁻_j is nonzero for λ > 0.

Substituting w = w⁺ − w⁻ yields the convex QP:

    minimize
        (1/2) (w⁺ − w⁻)^T (w⁺ − w⁻)
        + C * Σ_i ξ_i
        + λ 1^T (w⁺ + w⁻)
    subject to
        y_i ((w⁺ − w⁻)^T x_i + b) >= 1 - ξ_i
        w⁺ >= 0, w⁻ >= 0, ξ >= 0
        b free

Quadratic term in z = [w⁺; w⁻]:

    (1/2) [w⁺; w⁻]^T  [  I, -I ;  -I, I ]  [w⁺; w⁻]

which is positive semidefinite.  Linear term: q_{w⁺} = λ 1, q_{w⁻} = λ 1,
q_ξ = C (or per-sample C_i when class weights are used).

When class_weight is enabled, the slack penalty is C_i ξ_i with
C_i = C * n / (2 n_{y_i}), i.e. sklearn's "balanced" weights times C.
That is a training-only modelling choice; the test set is never reweighted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from module1.exceptions import InvalidHyperparameterError, OptimizationError


@dataclass
class QPData:
    P: sparse.csc_matrix
    q: np.ndarray
    A: sparse.csc_matrix
    l: np.ndarray
    u: np.ndarray
    n_features: int
    n_samples: int
    n_variables: int
    C_samples: np.ndarray
    y: np.ndarray
    X: np.ndarray
    lambda_l1: float
    C: float


def build_soft_margin_l1_qp(
    X: np.ndarray,
    y_pm1: np.ndarray,
    C: float,
    lambda_l1: float,
    sample_weights: np.ndarray | None = None,
) -> QPData:
    if C <= 0:
        raise InvalidHyperparameterError("C must be positive.")
    if lambda_l1 < 0:
        raise InvalidHyperparameterError("lambda_l1 must be non-negative.")
    if X.ndim != 2:
        raise OptimizationError("X must be a 2D array.")
    m, n = X.shape
    if m == 0 or n == 0:
        raise OptimizationError("Empty design matrix.")
    if y_pm1.shape != (m,):
        raise OptimizationError("y must have shape (n_samples,).")
    if not np.all(np.isin(y_pm1, [-1.0, 1.0])):
        raise OptimizationError("SVM labels must be in {-1, +1}.")
    if not np.isfinite(X).all():
        raise OptimizationError("Design matrix contains non-finite values.")

    if sample_weights is None:
        C_samples = np.full(m, C, dtype=np.float64)
    else:
        if sample_weights.shape != (m,):
            raise OptimizationError("sample_weights must match the number of training rows.")
        C_samples = (C * sample_weights).astype(np.float64)

    n_var = 2 * n + 1 + m
    # P = blkdiag( [I, -I; -I, I], 0 for b, 0 for ξ )
    I_n = sparse.eye(n, format="csc")
    P_w = sparse.bmat([[I_n, -I_n], [-I_n, I_n]], format="csc")
    P = sparse.block_diag((P_w, sparse.csc_matrix((1 + m, 1 + m))), format="csc")
    P = sparse.triu(P, format="csc").astype(np.float64)

    q = np.zeros(n_var, dtype=np.float64)
    q[:n] = lambda_l1
    q[n : 2 * n] = lambda_l1
    q[2 * n + 1 :] = C_samples

    y = y_pm1.astype(np.float64)
    YX = X * y[:, None]

    # Variable bounds encoded as identity rows: l <= z <= u
    A_bounds = sparse.eye(n_var, format="csc")
    l_bounds = np.full(n_var, 0.0)
    u_bounds = np.full(n_var, np.inf)
    l_bounds[2 * n] = -np.inf  # bias unconstrained

    # Margin: YX w⁺ − YX w⁻ + y b + ξ  >= 1
    A_margin = sparse.hstack(
        [
            sparse.csc_matrix(YX),
            sparse.csc_matrix(-YX),
            sparse.csc_matrix(y.reshape(-1, 1)),
            sparse.eye(m, format="csc"),
        ],
        format="csc",
    )
    l_margin = np.ones(m, dtype=np.float64)
    u_margin = np.full(m, np.inf)

    A = sparse.vstack([A_bounds, A_margin], format="csc").astype(np.float64)
    l = np.concatenate([l_bounds, l_margin])
    u = np.concatenate([u_bounds, u_margin])

    return QPData(
        P=P,
        q=q,
        A=A,
        l=l,
        u=u,
        n_features=n,
        n_samples=m,
        n_variables=n_var,
        C_samples=C_samples,
        y=y,
        X=X,
        lambda_l1=float(lambda_l1),
        C=float(C),
    )


def unpack_primal(z: np.ndarray, n_features: int, n_samples: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    n = n_features
    w_plus = np.asarray(z[:n], dtype=np.float64)
    w_minus = np.asarray(z[n : 2 * n], dtype=np.float64)
    bias = float(z[2 * n])
    xi = np.asarray(z[2 * n + 1 : 2 * n + 1 + n_samples], dtype=np.float64)
    return w_plus, w_minus, bias, xi


def primal_objective(w: np.ndarray, xi: np.ndarray, C_samples: np.ndarray, lambda_l1: float) -> float:
    return 0.5 * float(w @ w) + float(C_samples @ xi) + lambda_l1 * float(np.sum(np.abs(w)))
