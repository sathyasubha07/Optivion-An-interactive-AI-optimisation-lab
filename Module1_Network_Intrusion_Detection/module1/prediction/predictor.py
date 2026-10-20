"""Inference for new flows. Decision scores are not probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from module1.exceptions import PredictionInputError
from module1.models.schemas import PredictionRecord
from module1.preprocessing.pipeline import FittedPreprocessor


def scores_from_matrix(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return X @ weights + bias


def predict_labels(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(scores >= threshold, 1, 0).astype(np.int32)


def predict_records(
    X: np.ndarray,
    weights: np.ndarray,
    bias: float,
    feature_names: list[str],
    threshold: float,
    include_contributions: bool = False,
    top_k: int = 8,
    sparsity_threshold: float = 1e-6,
) -> list[PredictionRecord]:
    scores = scores_from_matrix(X, weights, bias)
    labels = predict_labels(scores, threshold)
    records: list[PredictionRecord] = []
    for i, score in enumerate(scores):
        contribs = None
        if include_contributions:
            raw = weights * X[i]
            order = np.argsort(-np.abs(raw))[: max(1, top_k)]
            contribs = []
            for idx in order:
                if abs(raw[idx]) < sparsity_threshold:
                    continue
                contribs.append(
                    {
                        "feature": feature_names[idx] if idx < len(feature_names) else f"f{idx}",
                        "contribution": float(raw[idx]),
                        "weight": float(weights[idx]),
                        "scaled_value": float(X[i, idx]),
                    }
                )
        records.append(
            PredictionRecord(
                predicted_label=int(labels[i]),
                predicted_class="Attack" if labels[i] == 1 else "Benign",
                decision_score=float(score),
                margin=float(abs(score)),
                on_margin=bool(abs(abs(score) - 1.0) <= 1e-2),
                feature_contributions=contribs,
            )
        )
    return records


def frame_from_samples(samples: list[dict], preprocessor: FittedPreprocessor) -> np.ndarray:
    if not samples:
        raise PredictionInputError("samples must be a non-empty list of feature dictionaries.")
    n_model = len(preprocessor.feature_names)
    if "vector" in samples[0]:
        matrix = np.asarray([row["vector"] for row in samples], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != n_model:
            raise PredictionInputError(
                f"vector length {matrix.shape[1] if matrix.ndim == 2 else 'invalid'} "
                f"does not match {n_model} model features. "
                "Pass a preprocessed vector, or raw named CIC-IDS2017 columns."
            )
        if not np.isfinite(matrix).all():
            raise PredictionInputError("vector samples contain non-finite values.")
        return matrix

    frame = pd.DataFrame(samples)
    missing = [c for c in preprocessor.source_feature_names if c not in frame.columns]
    if missing:
        raise PredictionInputError(
            "Each sample must include the original trained feature names, or a 'vector' "
            f"of length {n_model}. Missing: {missing[:8]}"
        )
    return preprocessor.transform(frame)
