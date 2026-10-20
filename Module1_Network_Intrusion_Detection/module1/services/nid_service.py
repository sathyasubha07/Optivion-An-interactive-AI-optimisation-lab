"""Orchestration for Module 1. The API and CLI call this, not the optimizer internals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from module1.analysis.kkt import analyze_kkt
from module1.analysis.sparsity import ranked_features, sparsity_report
from module1.analysis.support_vectors import analyze_support_vectors
from module1.config.settings import Module1Config, load_config
from module1.data.loader import LoadedDataset, binary_and_category, load_cicids2017, make_synthetic_flows
from module1.evaluation.metrics import evaluate_model
from module1.evaluation.threshold import select_decision_threshold, selection_score
from module1.exceptions import InsufficientSamplesError, ModelNotTrainedError, OptimizationError
from module1.experiments.baseline import comparison_row, default_grid
from module1.logging_setup import configure_logging, get_logger, log_kv
from module1.models.schemas import (
    DatasetSummary,
    ModelSummary,
    Module1Result,
    OptimizationReport,
    PredictionRecord,
)
from module1.optimization.formulation import build_soft_margin_l1_qp
from module1.optimization.solver import SolverResult, solve_qp
from module1.prediction.predictor import frame_from_samples, predict_records, scores_from_matrix
from module1.preprocessing.imbalance import apply_imbalance
from module1.preprocessing.pipeline import FittedPreprocessor, clean_frame, fit_preprocessor
from module1.preprocessing.split import stratified_split

logger = get_logger(__name__)

FORMULATION_TEXT = (
    "minimize 1/2||w||^2 + C Σ ξ_i + λ||w||_1  "
    "s.t. y_i(w^T x_i + b) >= 1 - ξ_i, ξ_i >= 0, "
    "with w = w^+ - w^-, w^+ >= 0, w^- >= 0, ||w||_1 = 1^T(w^+ + w^-)"
)


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return "unknown"


def software_versions() -> dict[str, str]:
    return {
        "module1": "0.1.0",
        "numpy": _pkg_version("numpy"),
        "pandas": _pkg_version("pandas"),
        "scikit-learn": _pkg_version("scikit-learn"),
        "scipy": _pkg_version("scipy"),
        "osqp": _pkg_version("osqp"),
        "fastapi": _pkg_version("fastapi"),
    }


def _binary_to_pm1(y: np.ndarray) -> np.ndarray:
    return np.where(y.astype(int) == 1, 1.0, -1.0)


def _stratified_subsample(frame: pd.DataFrame, label_column: str, size: int, seed: int) -> pd.DataFrame:
    if len(frame) <= size:
        return frame
    y, _ = binary_and_category(frame[label_column])
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    counts = {0: int(np.sum(y == 0)), 1: int(np.sum(y == 1))}
    target = {}
    for cls in (0, 1):
        prop = counts[cls] / len(frame)
        target[cls] = max(1, int(round(size * prop)))
    delta = size - sum(target.values())
    target[1] = max(1, target[1] + delta)
    taken = 0
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        k = min(len(idx), target[cls])
        chosen = rng.choice(idx, size=k, replace=False)
        keep.append(chosen)
        taken += k
    chosen_all = np.concatenate(keep)
    if taken > size:
        chosen_all = rng.choice(chosen_all, size=size, replace=False)
    return frame.iloc[np.sort(chosen_all)].reset_index(drop=True)


def _fit_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float,
    lambda_l1: float,
    sample_weights: np.ndarray | None,
    solver_cfg,
) -> tuple[SolverResult, Any]:
    qp = build_soft_margin_l1_qp(
        X_train,
        _binary_to_pm1(y_train),
        C=C,
        lambda_l1=lambda_l1,
        sample_weights=sample_weights,
    )
    solved = solve_qp(qp, solver_cfg)
    return solved, qp


@dataclass
class TrainedState:
    result: Module1Result
    weights: np.ndarray
    bias: float
    preprocessor: FittedPreprocessor
    config: Module1Config
    X_test: np.ndarray
    y_test: np.ndarray


class NIDService:
    """In-process Module 1 facade for API/CLI."""

    def __init__(self, config: Module1Config | None = None) -> None:
        self.base_config = config or load_config()
        configure_logging(self.base_config.logging.level)
        self.state: TrainedState | None = None
        self.status: Literal["idle", "training", "ready", "error"] = "idle"
        self.status_message = "No model trained yet."

    def train(
        self,
        dataset_source: Literal["cicids2017", "synthetic"] = "cicids2017",
        config_overrides: dict[str, Any] | None = None,
    ) -> Module1Result:
        self.status = "training"
        self.status_message = "Training in progress."
        try:
            result, trained = self._run_pipeline(dataset_source, config_overrides)
            self.state = trained
            self.status = "ready"
            self.status_message = (
                f"Model ready ({result.dataset.evaluation_protocol}, "
                f"source={result.dataset.source})."
            )
            return result
        except Exception as exc:
            self.status = "error"
            self.status_message = str(exc)
            logger.exception("Training failed")
            raise

    def _run_pipeline(
        self,
        dataset_source: Literal["cicids2017", "synthetic"],
        config_overrides: dict[str, Any] | None,
        lambda_override: float | None = None,
        C_override: float | None = None,
        reuse_split: dict[str, Any] | None = None,
        allow_tuning: bool = True,
    ) -> tuple[Module1Result, TrainedState]:
        cfg = self.base_config.merged(config_overrides)
        if C_override is not None:
            cfg.model.C = float(C_override)
        if lambda_override is not None:
            cfg.model.lambda_l1 = float(lambda_override)

        notes: list[str] = []
        if reuse_split is None:
            prepared = self._prepare_data(dataset_source, cfg)
            notes = list(prepared["notes"])
        else:
            prepared = reuse_split
            notes = list(prepared["notes"])

        label_col = prepared["label_col"]
        split = prepared["split"]
        imb = apply_imbalance(
            split.train,
            split.y_train,
            split.category_train,
            cfg.imbalance,
            cfg.random_seed,
        )
        feature_names = [c for c in imb.train_frame.columns if c != label_col]
        if len(feature_names) < 1:
            raise InsufficientSamplesError("No features left after cleaning.")

        preprocessor = fit_preprocessor(imb.train_frame, feature_names, cfg.preprocessing)
        feature_names = preprocessor.feature_names
        X_train = preprocessor.transform(imb.train_frame)
        y_train = imb.y_train
        X_test = preprocessor.transform(split.test)
        y_test = split.y_test
        if len(split.val):
            X_val = preprocessor.transform(split.val)
            y_val = split.y_val
        else:
            X_val = np.empty((0, X_train.shape[1]))
            y_val = np.array([], dtype=y_train.dtype)

        if len(np.unique(y_train)) < 2:
            raise InsufficientSamplesError("Training split must contain both classes.")

        sample_weights = imb.sample_weights if cfg.imbalance.method == "class_weight" else None
        tuned = False
        threshold_from_val = False
        selected_score: float | None = None
        C_use = cfg.model.C
        lam_use = cfg.model.lambda_l1
        threshold = cfg.model.decision_threshold

        do_tune = (
            allow_tuning
            and cfg.tuning.enabled
            and C_override is None
            and lambda_override is None
            and len(y_val) >= 20
            and len(np.unique(y_val)) == 2
        )
        if do_tune:
            C_use, lam_use, selected_score, tune_notes = self._select_hyperparameters(
                X_train,
                y_train,
                X_val,
                y_val,
                sample_weights,
                cfg,
            )
            tuned = True
            notes.extend(tune_notes)
            cfg.model.C = C_use
            cfg.model.lambda_l1 = lam_use
        elif cfg.tuning.enabled and allow_tuning and C_override is None:
            notes.append(
                "Hyperparameter tuning was skipped because the validation split was too small "
                "or lacked both classes. Config C and λ were used."
            )

        log_kv(logger, "training_start", samples=X_train.shape[0], features=X_train.shape[1], C=C_use, lambda_l1=lam_use)
        solved, qp = _fit_svm(X_train, y_train, C_use, lam_use, sample_weights, cfg.solver)

        if cfg.tuning.select_threshold and len(y_val) >= 20 and len(np.unique(y_val)) == 2:
            val_scores = scores_from_matrix(X_val, solved.w, solved.bias)
            chosen = select_decision_threshold(y_val, val_scores, fpr_penalty=cfg.tuning.fpr_penalty)
            threshold = chosen.threshold
            threshold_from_val = True
            selected_score = chosen.score if selected_score is None else selected_score
            notes.append(
                f"Decision threshold {threshold:.6g} was selected on the validation split "
                f"(attack_f1={chosen.attack_f1:.4f}, FPR={chosen.false_positive_rate:.4f}). "
                "The test split was not used."
            )
            cfg.model.decision_threshold = threshold
        else:
            notes.append(f"Using configured decision threshold {threshold}.")

        kkt = analyze_kkt(solved, qp, cfg.solver)
        log_kv(logger, "kkt_analysis", residual=kkt.kkt_residual, within=kkt.within_tolerance)
        sparse = sparsity_report(solved.w, cfg.model.sparsity_threshold)
        log_kv(
            logger,
            "sparsity",
            nonzero=sparse.nonzero_features,
            total=sparse.total_features,
            pct=round(sparse.sparsity_percentage, 2),
        )
        sv_report, _ = analyze_support_vectors(solved, qp, cfg.solver)
        evaluation = evaluate_model(X_test, y_test, solved.w, solved.bias, threshold)
        log_kv(
            logger,
            "evaluation_complete",
            accuracy=round(evaluation.accuracy, 4),
            attack_recall=round(evaluation.attack_recall, 4),
            f1=round(evaluation.f1, 4),
        )
        features = ranked_features(solved.w, feature_names, cfg.model.sparsity_threshold)

        dataset = DatasetSummary(
            name=prepared["name"],
            source=prepared["source"],
            evaluation_protocol=prepared["protocol"],
            n_rows_loaded=prepared["n_loaded"],
            n_rows_after_clean=prepared["n_clean"],
            train_samples=int(X_train.shape[0]),
            validation_samples=int(len(y_val)),
            test_samples=int(X_test.shape[0]),
            n_features=int(X_train.shape[1]),
            n_benign_loaded=prepared["n_benign_loaded"],
            n_attack_loaded=prepared["n_attack_loaded"],
            n_benign_train_before_balance=imb.n_benign_before,
            n_attack_train_before_balance=imb.n_attack_before,
            n_benign_train_after_balance=imb.n_benign_after,
            n_attack_train_after_balance=imb.n_attack_after,
            n_benign_test=int(np.sum(y_test == 0)),
            n_attack_test=int(np.sum(y_test == 1)),
            files=prepared["files"],
            notes=notes + preprocessor.notes,
        )
        model = ModelSummary(
            formulation=FORMULATION_TEXT,
            C=C_use,
            lambda_l1=lam_use,
            decision_threshold=threshold,
            sparsity_threshold=cfg.model.sparsity_threshold,
            random_seed=cfg.random_seed,
            class_weight_applied=cfg.imbalance.method == "class_weight",
            hyperparameters_tuned=tuned,
            threshold_selected_on_validation=threshold_from_val,
            validation_samples=int(len(y_val)),
            selection_score=selected_score,
        )
        optimization = OptimizationReport(
            status=solved.status,
            status_val=solved.status_val,
            converged=solved.converged,
            objective_value=solved.objective_value,
            primal_objective=solved.primal_objective,
            iterations=solved.iterations,
            training_time_seconds=solved.training_time_seconds,
            polish_status=solved.polish_status,
            primal_residual=solved.primal_residual,
            dual_residual=solved.dual_residual,
            run_time=solved.run_time,
            notes=solved.notes,
        )
        result = Module1Result(
            dataset=dataset,
            model=model,
            optimization=optimization,
            kkt=kkt,
            sparsity=sparse,
            support_vectors=sv_report,
            evaluation=evaluation,
            features=features,
            weights=solved.w.astype(float).tolist(),
            bias=float(solved.bias),
            software=software_versions(),
        )
        trained = TrainedState(
            result=result,
            weights=solved.w,
            bias=solved.bias,
            preprocessor=preprocessor,
            config=cfg,
            X_test=X_test,
            y_test=y_test,
        )
        trained._reuse = prepared  # type: ignore[attr-defined]
        return result, trained

    def _select_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray | None,
        cfg: Module1Config,
    ) -> tuple[float, float, float, list[str]]:
        pairs = [(c, lam) for c in cfg.tuning.C_values for lam in cfg.tuning.lambda_values]
        best: tuple[float, float, float] | None = None
        notes = [
            f"Selected C and λ on the validation split ({len(y_val)} rows) using "
            f"score = attack_f1 - {cfg.tuning.fpr_penalty} * FPR. Test data was not used."
        ]
        failures = 0
        for c, lam in pairs:
            try:
                solved, _ = _fit_svm(X_train, y_train, c, lam, sample_weights, cfg.solver)
            except OptimizationError as exc:
                failures += 1
                log_kv(logger, "tuning_cell_failed", C=c, lambda_l1=lam, error=str(exc))
                continue
            val_eval = evaluate_model(X_val, y_val, solved.w, solved.bias, 0.0)
            metric = selection_score(val_eval.attack_f1, val_eval.false_positive_rate, cfg.tuning.fpr_penalty)
            log_kv(
                logger,
                "tuning_cell",
                C=c,
                lambda_l1=lam,
                val_attack_f1=round(val_eval.attack_f1, 4),
                val_fpr=round(val_eval.false_positive_rate, 4),
                score=round(metric, 4),
            )
            if best is None or metric > best[2]:
                best = (c, lam, metric)
        if best is None:
            raise OptimizationError("All hyperparameter grid cells failed to solve.")
        notes.append(
            f"Validation-selected C={best[0]}, lambda_l1={best[1]} "
            f"(score={best[2]:.4f}; {failures} grid failures)."
        )
        return best[0], best[1], best[2], notes

    def _prepare_data(self, dataset_source: str, cfg: Module1Config) -> dict[str, Any]:
        loaded = self._load(dataset_source, cfg)
        cleaned, clean_notes = clean_frame(loaded, cfg.data.drop_duplicates)
        notes = list(clean_notes)
        protocol: Literal["development_sample", "configured_dataset"]
        if dataset_source == "synthetic":
            protocol = "development_sample"
            notes.append("SYNTHETIC data was used. These metrics are not CIC-IDS2017 results.")
        elif cfg.data.use_development_sample:
            protocol = "development_sample"
            before = len(cleaned)
            cleaned = _stratified_subsample(
                cleaned,
                loaded.label_column,
                cfg.data.development_sample_size,
                cfg.random_seed,
            )
            notes.append(
                f"DEVELOPMENT SAMPLE: stratified subsample {before} -> {len(cleaned)} rows. "
                "Do not treat this as a full CIC-IDS2017 evaluation."
            )
        else:
            protocol = "configured_dataset"
            if cfg.data.max_rows is not None:
                notes.append(
                    f"configured_dataset with data.max_rows={cfg.data.max_rows} "
                    "(row cap applied at load time, distributed across CSV files)."
                )

        split = stratified_split(cleaned, loaded.label_column, cfg.split, cfg.random_seed)
        n_benign_loaded = int((binary_and_category(cleaned[loaded.label_column])[0] == 0).sum())
        n_attack_loaded = int(len(cleaned) - n_benign_loaded)
        return {
            "protocol": protocol,
            "split": split,
            "n_benign_loaded": n_benign_loaded,
            "n_attack_loaded": n_attack_loaded,
            "files": loaded.files,
            "n_loaded": loaded.n_rows_loaded,
            "n_clean": len(cleaned),
            "source": dataset_source,
            "name": "synthetic_flows" if dataset_source == "synthetic" else "CIC-IDS2017",
            "label_col": loaded.label_column,
            "notes": notes,
            "loaded": loaded,
        }

    def _load(self, dataset_source: str, cfg: Module1Config) -> LoadedDataset:
        if dataset_source == "synthetic":
            n = min(800, max(cfg.data.min_samples, 300))
            log_kv(logger, "loading_synthetic", n=n, seed=cfg.random_seed)
            return make_synthetic_flows(n_samples=n, n_features=12, seed=cfg.random_seed)
        log_kv(logger, "loading_cicids2017", directory=str(cfg.data.resolved_directory()))
        return load_cicids2017(cfg.data)

    def require_state(self) -> TrainedState:
        if self.state is None:
            raise ModelNotTrainedError()
        return self.state

    def predict(
        self,
        samples: list[dict[str, Any]],
        include_contributions: bool = True,
        top_k: int = 8,
    ) -> list[PredictionRecord]:
        st = self.require_state()
        X = frame_from_samples(samples, st.preprocessor)
        return predict_records(
            X,
            st.weights,
            st.bias,
            st.preprocessor.feature_names,
            st.config.model.decision_threshold,
            include_contributions=include_contributions,
            top_k=top_k,
            sparsity_threshold=st.config.model.sparsity_threshold,
        )

    def evaluate_current(self) -> Module1Result:
        return self.require_state().result

    def experiment(
        self,
        dataset_source: Literal["cicids2017", "synthetic"] = "cicids2017",
        compare_baseline: bool = True,
        C_values: list[float] | None = None,
        lambda_values: list[float] | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fair comparison on one shared split. Does not claim L1 is better."""
        self.status = "training"
        self.status_message = "Running experiment."
        try:
            primary, trained = self._run_pipeline(dataset_source, config_overrides)
            reuse = trained._reuse  # type: ignore[attr-defined]
            rows = [comparison_row(primary)]
            baseline = None
            if compare_baseline and primary.model.lambda_l1 != 0.0:
                base_res, _ = self._run_pipeline(
                    dataset_source,
                    config_overrides,
                    lambda_override=0.0,
                    C_override=primary.model.C,
                    reuse_split=reuse,
                    allow_tuning=False,
                )
                baseline = comparison_row(base_res)
                rows.append(baseline)

            grid_rows: list[dict[str, Any]] = []
            Cs, lams = default_grid(trained.config)
            if C_values:
                Cs = C_values
            if lambda_values:
                lams = lambda_values
            pairs = [(c, l) for c in Cs for l in lams][:9]
            for c, lam in pairs:
                if abs(c - primary.model.C) < 1e-15 and abs(lam - primary.model.lambda_l1) < 1e-15:
                    grid_rows.append(comparison_row(primary))
                    continue
                res, _ = self._run_pipeline(
                    dataset_source,
                    config_overrides,
                    lambda_override=lam,
                    C_override=c,
                    reuse_split=reuse,
                    allow_tuning=False,
                )
                grid_rows.append(comparison_row(res))

            self.state = trained
            self.status = "ready"
            self.status_message = "Experiment finished; primary model retained."
            return {
                "module": "network_intrusion_detection",
                "note": (
                    "Primary C/λ/threshold were selected on validation only. "
                    "Grid rows reuse that split; their test metrics are descriptive, "
                    "not a second selection loop. L1 is not automatically better."
                ),
                "primary": comparison_row(primary),
                "baseline_lambda_0": baseline,
                "grid": grid_rows,
                "full_primary_result": primary.model_dump(),
            }
        except Exception:
            self.status = "error"
            logger.exception("Experiment failed")
            raise


_SERVICE: NIDService | None = None


def get_service() -> NIDService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = NIDService()
    return _SERVICE


def reset_service(config: Module1Config | None = None) -> NIDService:
    global _SERVICE
    _SERVICE = NIDService(config)
    return _SERVICE
