"""Held-out evaluation. Attack is the positive class (label 1)."""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from module1.models.schemas import ConfusionCounts, EvaluationReport
from module1.prediction.predictor import predict_labels, scores_from_matrix


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num / den)


def evaluate_binary(
    y_true: np.ndarray,
    scores: np.ndarray,
    decision_threshold: float,
    inference_seconds: float,
) -> EvaluationReport:
    y_pred = predict_labels(scores, decision_threshold)
    y_true = y_true.astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    n = int(len(y_true))
    accuracy = _safe_div(tp + tn, n)
    attack_precision = _safe_div(tp, tp + fp)
    attack_recall = _safe_div(tp, tp + fn)
    attack_f1 = _safe_div(2 * attack_precision * attack_recall, attack_precision + attack_recall)
    benign_precision = _safe_div(tn, tn + fn)
    benign_recall = _safe_div(tn, tn + fp)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    notes = [
        "Metrics are computed on the held-out test split only.",
        "Attack is the positive class. High accuracy with low attack recall is not a success.",
        "sklearn precision/recall/f1 default to the positive class (Attack) and match attack_*.",
    ]
    return EvaluationReport(
        n_test=n,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        attack_precision=attack_precision,
        attack_recall=attack_recall,
        attack_f1=attack_f1,
        benign_precision=benign_precision,
        benign_recall=benign_recall,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        confusion_matrix=ConfusionCounts(
            true_negative=tn,
            false_positive=fp,
            false_negative=fn,
            true_positive=tp,
        ),
        inference_time_seconds=float(inference_seconds),
        inference_time_per_sample_seconds=float(inference_seconds / n) if n else 0.0,
        notes=notes,
    )


def evaluate_model(X_test: np.ndarray, y_test: np.ndarray, weights: np.ndarray, bias: float, threshold: float) -> EvaluationReport:
    t0 = time.perf_counter()
    scores = scores_from_matrix(X_test, weights, bias)
    elapsed = time.perf_counter() - t0
    return evaluate_binary(y_test, scores, threshold, elapsed)
