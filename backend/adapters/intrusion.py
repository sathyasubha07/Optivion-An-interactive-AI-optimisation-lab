"""Adapter for Module 1 — NIDService.train() + predict_records(). Module source is not modified."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from backend.adapters.base import Adapter, RunContext, Sample, metric
from backend.jsonutil import NOT_COMPUTED, download, maybe_bool, maybe_float, maybe_int, maybe_text
from backend.paths import MODULE1_ROOT

SAMPLE_ID = "synthetic-flows"
# Module 1's NIDService._load("synthetic") generates n = min(800, max(min_samples, 300)) flows
# with 12 features and seed = config.random_seed (42 in default.yaml). With min_samples=30 that
# is 300 rows, so the downloadable sample CSV below is exactly the data the sample run trains on.
SAMPLE_ROWS = 300
SAMPLE_FEATURES = 12
SAMPLE_SEED = 42

UPLOAD_OVERRIDES = {
    "min_samples": 30,
    "use_development_sample": True,
    "development_sample_size": 3000,
    "max_rows": 20000,
}


def _ensure_path() -> None:
    root = str(MODULE1_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def list_samples() -> list[Sample]:
    return [
        Sample(
            id=SAMPLE_ID,
            label="Synthetic flows",
            description=(
                f"{SAMPLE_ROWS} synthetic flows × {SAMPLE_FEATURES} features from Module 1's "
                "make_synthetic_flows(). Not CIC-IDS2017 — no real traffic dataset is bundled."
            ),
        )
    ]


def make_sample(sample_id: str, dest_dir: Path) -> Path:
    if sample_id != SAMPLE_ID:
        raise ValueError(f"Unknown sample '{sample_id}'.")
    _ensure_path()
    from module1.data.loader import make_synthetic_flows

    dest_dir.mkdir(parents=True, exist_ok=True)
    loaded = make_synthetic_flows(n_samples=SAMPLE_ROWS, n_features=SAMPLE_FEATURES, seed=SAMPLE_SEED)
    dest = dest_dir / "synthetic_flows.csv"
    loaded.frame.to_csv(dest, index=False)
    return dest


def _fmt(value: Any, spec: str = ".4f") -> str:
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return NOT_COMPUTED


def _explanation(result, upload_note: str) -> str:
    ev, opt, kkt, sparse, sv, ds, model = (
        result.evaluation, result.optimization, result.kkt, result.sparsity,
        result.support_vectors, result.dataset, result.model,
    )
    cm = ev.confusion_matrix
    top = [f for f in result.features if f.retained_by_l1][:5]
    top_lines = [f"  {f.rank:>2}. {f.name:<32} w = {f.weight:+.4f}" for f in top] or ["  (no non-zero weights)"]
    notes = [f"  - {n}" for n in ds.notes] or ["  - none"]
    return "\n".join([
        "OPTIVISION — MODULE 1: NETWORK INTRUSION DETECTION",
        "=" * 60,
        "",
        "This text is assembled by the OptiVision adapter from Module 1's own result",
        "object (the module has no text-explanation function). Every number below",
        "comes from that object.",
        "",
        "WHAT WAS SOLVED",
        "  A soft-margin linear SVM with an L1 penalty, written as a convex QP by",
        "  splitting w = w+ - w-, and solved with OSQP (an ADMM-based QP solver).",
        f"  {model.formulation}",
        "",
        "DATA",
        f"  {upload_note}",
        f"  Dataset / source : {ds.name} / {ds.source} ({ds.evaluation_protocol})",
        f"  Rows train / validation / test : {ds.train_samples} / {ds.validation_samples} / {ds.test_samples}",
        f"  Features after cleaning        : {ds.n_features}",
        "",
        "MODEL CHOICES",
        f"  C = {model.C}, lambda_L1 = {model.lambda_l1}, decision threshold = {_fmt(model.decision_threshold, '.6g')}",
        f"  Hyperparameters tuned on validation : {model.hyperparameters_tuned}",
        f"  Threshold selected on validation    : {model.threshold_selected_on_validation}",
        "",
        "OPTIMISATION",
        f"  OSQP status {opt.status!r}, converged = {opt.converged}, {opt.iterations} iterations",
        f"  Primal objective {_fmt(opt.primal_objective, '.6g')}; primal residual "
        f"{_fmt(opt.primal_residual, '.3e')}; dual residual {_fmt(opt.dual_residual, '.3e')}",
        f"  KKT residual {_fmt(kkt.kkt_residual, '.3e')} (within tolerance {kkt.tolerance:g}: {kkt.within_tolerance})",
        "",
        "WHAT THE MODEL USES",
        f"  {sparse.nonzero_features} of {sparse.total_features} feature weights are non-zero "
        f"({sparse.sparsity_percentage:.2f}% sparse). Largest |w|:",
        *top_lines,
        f"  {sv.count} support vectors ({100 * sv.fraction_of_training:.2f}% of training rows) define the margin.",
        "",
        "HOW WELL IT DETECTS (held-out test split of this run)",
        f"  Accuracy {ev.accuracy:.4f} | Precision {ev.precision:.4f} | Recall {ev.recall:.4f} | F1 {ev.f1:.4f}",
        f"  Attack recall {ev.attack_recall:.4f} | False-positive rate {ev.false_positive_rate:.4f}",
        f"  Confusion: TN {cm.true_negative}, FP {cm.false_positive}, FN {cm.false_negative}, TP {cm.true_positive}",
        "",
        "HOW TO READ A PREDICTION",
        "  Score = w^T x + b. Score >= threshold means Attack. It is a signed margin,",
        "  not a probability. Per-feature contributions are w_j * x_j on scaled features.",
        "",
        "MODULE NOTES",
        *notes,
    ])


def run(ctx: RunContext) -> dict[str, Any]:
    _ensure_path()
    from module1.prediction.predictor import predict_records
    from module1.services.nid_service import NIDService

    service = NIDService()
    if ctx.sample_id == SAMPLE_ID:
        result = service.train(dataset_source="synthetic", config_overrides={"data": {"min_samples": 30}})
        upload_note = (
            "Sample run: Module 1's synthetic flows (the same rows as the downloadable sample CSV). "
            "These metrics are not CIC-IDS2017 results."
        )
    else:
        overrides = {
            "data": {
                **UPLOAD_OVERRIDES,
                "directory": str(ctx.input_path.resolve().parent),
                "file_glob": ctx.input_path.name,
            }
        }
        result = service.train(dataset_source="cicids2017", config_overrides=overrides)
        upload_note = (
            f"Trained and evaluated on your file '{ctx.input_path.name}' through Module 1's CIC-IDS2017 "
            "CSV loader (a 'Label' column with BENIGN / attack names is required). Metrics are for this "
            "file's held-out split, not a published benchmark score."
        )

    state = service.require_state()
    records = predict_records(
        state.X_test, state.weights, state.bias, state.preprocessor.feature_names,
        state.config.model.decision_threshold, include_contributions=True, top_k=3,
        sparsity_threshold=state.config.model.sparsity_threshold,
    )
    pred_rows = []
    for i, (rec, true_label) in enumerate(zip(records, state.y_test.tolist())):
        contribs = rec.feature_contributions or []
        pred_rows.append({
            "row": i,
            "true_class": "Attack" if int(true_label) == 1 else "Benign",
            "predicted_class": rec.predicted_class,
            "correct": rec.predicted_label == int(true_label),
            "decision_score": rec.decision_score,
            "top_features": ", ".join(f"{c['feature']} ({c['contribution']:+.3f})" for c in contribs),
        })

    ev, opt, kkt, sparse, sv, ds, model = (
        result.evaluation, result.optimization, result.kkt, result.sparsity,
        result.support_vectors, result.dataset, result.model,
    )
    cm = ev.confusion_matrix
    explanation = _explanation(result, upload_note)

    out = ctx.out_dir
    (out / "explanation.txt").write_text(explanation, encoding="utf-8")
    pd.DataFrame(pred_rows).to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame([f.model_dump() for f in result.features]).to_csv(out / "feature_weights.csv", index=False)
    (out / "module1_result.json").write_text(json.dumps(result.model_dump(), indent=2, default=str), encoding="utf-8")

    features_sorted = sorted(result.features, key=lambda f: -f.abs_weight)

    return {
        "summary": [
            metric("attack_recall", "Attack recall", maybe_float(ev.attack_recall), "pct", higher_is_better=True,
                   hint="Share of attack flows in the test split that were caught."),
            metric("false_positive_rate", "False-positive rate", maybe_float(ev.false_positive_rate), "pct",
                   higher_is_better=False),
            metric("nonzero", "Features used", maybe_int(sparse.nonzero_features), "int",
                   hint=f"Non-zero weights out of {sparse.total_features} after L1."),
            metric("kkt_residual", "KKT residual", maybe_float(kkt.kkt_residual), "sci",
                   hint=f"Within tolerance {kkt.tolerance:g}: {'yes' if kkt.within_tolerance else 'no'}."),
        ],
        "metric_groups": [
            {
                "title": "Detection (held-out test split)",
                "note": f"{ev.n_test} test rows never used for training, tuning, or threshold selection.",
                "metrics": [
                    metric("accuracy", "Accuracy", maybe_float(ev.accuracy), "pct"),
                    metric("precision", "Precision", maybe_float(ev.precision), "pct"),
                    metric("recall", "Recall", maybe_float(ev.recall), "pct"),
                    metric("f1", "F1", maybe_float(ev.f1), "pct"),
                    metric("attack_f1", "Attack F1", maybe_float(ev.attack_f1), "pct"),
                    metric("false_negative_rate", "False-negative rate", maybe_float(ev.false_negative_rate), "pct"),
                ],
            },
            {
                "title": "Optimisation (OSQP)",
                "metrics": [
                    metric("status", "Solver status", maybe_text(opt.status), "text"),
                    metric("converged", "Converged", maybe_bool(opt.converged), "bool"),
                    metric("iterations", "ADMM iterations", maybe_int(opt.iterations), "int"),
                    metric("primal_objective", "Primal objective", maybe_float(opt.primal_objective), "sci"),
                    metric("primal_residual", "Primal residual", maybe_float(opt.primal_residual), "sci"),
                    metric("dual_residual", "Dual residual", maybe_float(opt.dual_residual), "sci"),
                    metric("training_time", "Solve time", maybe_float(opt.training_time_seconds), "float4", unit="s"),
                ],
            },
            {
                "title": "KKT conditions",
                "note": "Computed by Module 1 from OSQP's primal and dual solution.",
                "metrics": [
                    metric("kkt_residual", "KKT residual", maybe_float(kkt.kkt_residual), "sci"),
                    metric("within_tolerance", f"Within tolerance ({kkt.tolerance:g})", maybe_bool(kkt.within_tolerance), "bool"),
                    metric("stationarity", "Stationarity", maybe_float(kkt.stationarity_residual), "sci"),
                    metric("complementarity", "Complementary slackness", maybe_float(kkt.complementary_slackness), "sci"),
                    metric("primal_feasibility", "Primal feasibility", maybe_float(kkt.primal_feasibility), "sci"),
                    metric("dual_feasibility", "Dual feasibility", maybe_float(kkt.dual_feasibility), "sci"),
                ],
            },
            {
                "title": "Sparsity and support vectors",
                "metrics": [
                    metric("nonzero", "Non-zero weights", maybe_int(sparse.nonzero_features), "int"),
                    metric("total_features", "Total features", maybe_int(sparse.total_features), "int"),
                    metric("sparsity_pct", "Sparsity", maybe_float(sparse.sparsity_percentage / 100.0), "pct"),
                    metric("sv_count", "Support vectors", maybe_int(sv.count), "int"),
                    metric("sv_fraction", "Support-vector share", maybe_float(sv.fraction_of_training), "pct"),
                ],
            },
            {
                "title": "Model and data",
                "metrics": [
                    metric("C", "C (soft margin)", maybe_float(model.C), "float4"),
                    metric("lambda_l1", "λ (L1 weight)", maybe_float(model.lambda_l1), "float4"),
                    metric("threshold", "Decision threshold", maybe_float(model.decision_threshold), "float4"),
                    metric("tuned", "C, λ tuned on validation", maybe_bool(model.hyperparameters_tuned), "bool"),
                    metric("train", "Training rows", maybe_int(ds.train_samples), "int"),
                    metric("val", "Validation rows", maybe_int(ds.validation_samples), "int"),
                    metric("test", "Test rows", maybe_int(ds.test_samples), "int"),
                ],
            },
        ],
        "replay": {
            "available": False,
            "note": (
                f"OSQP ran {opt.iterations} ADMM iterations inside its compiled solver. Module 1 records only "
                "the final iterate and residuals, not per-iteration values, so there is no recorded history to "
                "replay. The final solver diagnostics are shown under Results."
            ),
            "x_label": "ADMM iteration",
            "series": [],
        },
        "charts": [
            {
                "type": "bar",
                "key": "weights",
                "title": "Learned feature weights w",
                "x_label": "Feature",
                "y_label": "Weight",
                "data": [{"label": f.name, "value": f.weight} for f in features_sorted[:30]],
                "note": (
                    "Positive weights push a flow towards Attack, negative towards Benign. "
                    "Zero bars were removed by the L1 penalty. Association, not causation."
                ),
            }
        ],
        "tables": [
            {
                "key": "confusion",
                "title": "Confusion matrix (test split)",
                "columns": [
                    {"key": "actual", "label": "Actual \\ Predicted"},
                    {"key": "benign", "label": "Benign", "fmt": "int"},
                    {"key": "attack", "label": "Attack", "fmt": "int"},
                ],
                "rows": [
                    {"actual": "Benign", "benign": cm.true_negative, "attack": cm.false_positive},
                    {"actual": "Attack", "benign": cm.false_negative, "attack": cm.true_positive},
                ],
            },
            {
                "key": "features",
                "title": "Feature ranking",
                "columns": [
                    {"key": "rank", "label": "#", "fmt": "int"},
                    {"key": "name", "label": "Feature"},
                    {"key": "weight", "label": "Weight", "fmt": "float4"},
                    {"key": "retained_by_l1", "label": "Kept by L1", "fmt": "bool"},
                ],
                "rows": [f.model_dump(include={"rank", "name", "weight", "retained_by_l1"}) for f in result.features],
            },
            {
                "key": "predictions",
                "title": "Test-split predictions",
                "note": "Decision score = wᵀx + b (a signed margin, not a probability). First 50 rows; full list in downloads.",
                "columns": [
                    {"key": "row", "label": "Row", "fmt": "int"},
                    {"key": "true_class", "label": "Actual"},
                    {"key": "predicted_class", "label": "Predicted"},
                    {"key": "correct", "label": "Correct", "fmt": "bool"},
                    {"key": "decision_score", "label": "Score", "fmt": "float4"},
                    {"key": "top_features", "label": "Top contributions"},
                ],
                "rows": pred_rows[:50],
            },
        ],
        "explanation": explanation,
        "explanation_source": "Assembled by the OptiVision adapter from Module 1's result object.",
        "notes": [upload_note, *ds.notes, *kkt.limitations],
        "downloads": [
            d for d in (
                download("Explanation (TXT)", out / "explanation.txt"),
                download("Test predictions (CSV)", out / "test_predictions.csv"),
                download("Feature weights (CSV)", out / "feature_weights.csv"),
                download("Full Module 1 result (JSON)", out / "module1_result.json"),
            ) if d
        ],
    }


ADAPTER = Adapter(run=run, list_samples=list_samples, make_sample=make_sample)
