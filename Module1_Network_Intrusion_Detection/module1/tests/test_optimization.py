import numpy as np

from module1.analysis.kkt import analyze_kkt
from module1.analysis.sparsity import sparsity_report
from module1.config.settings import SolverConfig
from module1.optimization.formulation import build_soft_margin_l1_qp, unpack_primal
from module1.optimization.solver import solve_qp
from module1.prediction.predictor import predict_labels, scores_from_matrix
from module1.evaluation.metrics import evaluate_binary


def _separable_xy(n=60, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    base_w = [1.5, -1.0, 0.0, 0.0, 0.8]
    if d <= len(base_w):
        w = np.array(base_w[:d])
    else:
        w = np.zeros(d)
        w[: len(base_w)] = base_w
    y = np.where(X @ w + 0.1 > 0, 1.0, -1.0)
    # keep both classes
    if len(np.unique(y)) < 2:
        y[: n // 2] = -1
        y[n // 2 :] = 1
    return X, y


def test_l1_split_dimensions():
    X, y = _separable_xy()
    qp = build_soft_margin_l1_qp(X, y, C=1.0, lambda_l1=0.05)
    assert qp.n_variables == 2 * X.shape[1] + 1 + X.shape[0]
    assert qp.P.shape[0] == qp.n_variables
    w_plus, w_minus, b, xi = unpack_primal(np.zeros(qp.n_variables), qp.n_features, qp.n_samples)
    assert w_plus.shape == (X.shape[1],)
    assert xi.shape == (X.shape[0],)


def test_solver_and_kkt_on_synthetic():
    X, y = _separable_xy(n=80, d=6, seed=7)
    qp = build_soft_margin_l1_qp(X, y, C=1.0, lambda_l1=0.05)
    cfg = SolverConfig(max_iter=15000, eps_abs=1e-4, eps_rel=1e-4, kkt_tolerance=5.0)
    result = solve_qp(qp, cfg)
    assert result.converged
    assert np.isfinite(result.w).all()
    kkt = analyze_kkt(result, qp, cfg)
    assert kkt.primal_feasibility < 1.0
    assert kkt.kkt_residual >= 0
    assert kkt.stationarity_residual is not None


def test_lambda_increases_sparsity():
    X, y = _separable_xy(n=100, d=10, seed=8)
    dense = solve_qp(build_soft_margin_l1_qp(X, y, C=1.0, lambda_l1=0.0), SolverConfig())
    sparse = solve_qp(build_soft_margin_l1_qp(X, y, C=1.0, lambda_l1=0.5), SolverConfig())
    d_rep = sparsity_report(dense.w, 1e-4)
    s_rep = sparsity_report(sparse.w, 1e-4)
    assert s_rep.zero_features >= d_rep.zero_features


def test_prediction_and_metrics():
    X, y = _separable_xy(n=70, d=5, seed=9)
    qp = build_soft_margin_l1_qp(X, y, C=1.0, lambda_l1=0.02)
    result = solve_qp(qp, SolverConfig())
    scores = scores_from_matrix(X, result.w, result.bias)
    y_bin = np.where(y > 0, 1, 0)
    pred = predict_labels(scores, 0.0)
    assert set(pred.tolist()).issubset({0, 1})
    report = evaluate_binary(y_bin, scores, 0.0, inference_seconds=0.01)
    assert 0.0 <= report.accuracy <= 1.0
    assert report.confusion_matrix.true_positive + report.confusion_matrix.false_negative == int((y_bin == 1).sum())
    # decision scores must not be advertised as probabilities
    assert not np.all((scores >= 0) & (scores <= 1))
