"""Optional baseline (λ=0) and small C/λ grids. Same split protocol; measured numbers only."""

from __future__ import annotations

from typing import Any

from module1.config.settings import Module1Config


def comparison_row(result) -> dict[str, Any]:
    ev = result.evaluation
    sp = result.sparsity
    return {
        "C": result.model.C,
        "lambda_l1": result.model.lambda_l1,
        "decision_threshold": result.model.decision_threshold,
        "accuracy": ev.accuracy,
        "precision": ev.precision,
        "recall": ev.recall,
        "f1": ev.f1,
        "attack_precision": ev.attack_precision,
        "attack_recall": ev.attack_recall,
        "attack_f1": ev.attack_f1,
        "false_positive_rate": ev.false_positive_rate,
        "false_negative_rate": ev.false_negative_rate,
        "n_features": sp.total_features,
        "nonzero_features": sp.nonzero_features,
        "sparsity_percentage": sp.sparsity_percentage,
        "training_time_seconds": result.optimization.training_time_seconds,
        "solver_status": result.optimization.status,
        "evaluation_protocol": result.dataset.evaluation_protocol,
        "hyperparameters_tuned": result.model.hyperparameters_tuned,
        "threshold_selected_on_validation": result.model.threshold_selected_on_validation,
        "selection_score": result.model.selection_score,
    }


def default_grid(config: Module1Config) -> tuple[list[float], list[float]]:
    return list(config.tuning.C_values), list(config.tuning.lambda_values)
