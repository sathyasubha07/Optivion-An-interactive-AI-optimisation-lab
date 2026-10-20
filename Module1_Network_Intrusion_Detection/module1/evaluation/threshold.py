"""Validation-only decision-threshold selection.

The SVM score is w^T x + b.  The default cut is 0.  A different cut can
improve attack F1 / FPR on imbalanced data, but it must be chosen on
validation scores only — never on the test set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from module1.evaluation.metrics import evaluate_binary


@dataclass
class ThresholdSelection:
    threshold: float
    score: float
    attack_f1: float
    attack_recall: float
    false_positive_rate: float
    n_candidates: int


def selection_score(attack_f1: float, false_positive_rate: float, fpr_penalty: float) -> float:
    """Higher is better. Penalises false positives so accuracy-only winners lose."""
    return float(attack_f1) - float(fpr_penalty) * float(false_positive_rate)


def select_decision_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    fpr_penalty: float = 0.25,
    n_quantiles: int = 51,
) -> ThresholdSelection:
    candidates = [0.0]
    if scores.size:
        qs = np.quantile(scores, np.linspace(0.0, 1.0, n_quantiles))
        candidates.extend(float(q) for q in qs)
        candidates.extend([float(np.min(scores)), float(np.max(scores))])
    unique = np.unique(np.asarray(candidates, dtype=np.float64))

    best: ThresholdSelection | None = None
    for thr in unique:
        report = evaluate_binary(y_true, scores, float(thr), inference_seconds=0.0)
        metric = selection_score(report.attack_f1, report.false_positive_rate, fpr_penalty)
        current = ThresholdSelection(
            threshold=float(thr),
            score=metric,
            attack_f1=report.attack_f1,
            attack_recall=report.attack_recall,
            false_positive_rate=report.false_positive_rate,
            n_candidates=int(unique.size),
        )
        if best is None or metric > best.score + 1e-15:
            best = current
        elif abs(metric - best.score) <= 1e-15:
            if report.attack_recall > best.attack_recall + 1e-15:
                best = current
            elif abs(report.attack_recall - best.attack_recall) <= 1e-15 and report.false_positive_rate < best.false_positive_rate:
                best = current
    assert best is not None
    return best
