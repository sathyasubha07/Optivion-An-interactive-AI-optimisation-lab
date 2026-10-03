"""L1 sparsity and linear-model feature weights.

A coordinate is treated as zero when |w_j| <= sparsity_threshold.
The default threshold is 1e-6 (see config model.sparsity_threshold).

Weights are model-level contributions.  A large |w_j| does not prove that
feature j causes attacks; it means the fitted linear detector uses it.
"""

from __future__ import annotations

import numpy as np

from module1.models.schemas import FeatureWeight, SparsityReport


def sparsity_report(weights: np.ndarray, threshold: float) -> SparsityReport:
    total = int(weights.size)
    nonzero = int(np.sum(np.abs(weights) > threshold))
    zero = total - nonzero
    pct = 100.0 * zero / total if total else 0.0
    return SparsityReport(
        total_features=total,
        nonzero_features=nonzero,
        zero_features=zero,
        sparsity_percentage=float(pct),
        threshold=float(threshold),
        l1_norm=float(np.sum(np.abs(weights))),
        l2_norm=float(np.linalg.norm(weights)),
    )


def ranked_features(weights: np.ndarray, names: list[str], threshold: float) -> list[FeatureWeight]:
    order = np.argsort(-np.abs(weights))
    ranked: list[FeatureWeight] = []
    for rank, idx in enumerate(order, start=1):
        w = float(weights[idx])
        if w > threshold:
            meaning = (
                "Positive weight: increasing this (scaled) feature increases the Attack-side "
                "decision score. This is a model weight, not a causal claim."
            )
        elif w < -threshold:
            meaning = (
                "Negative weight: increasing this (scaled) feature decreases the Attack-side "
                "decision score. This is a model weight, not a causal claim."
            )
        else:
            meaning = "Weight is below the sparsity threshold; L1 treated it as unused."
        ranked.append(
            FeatureWeight(
                rank=rank,
                name=names[idx] if idx < len(names) else f"feature_{idx}",
                weight=w,
                abs_weight=abs(w),
                retained_by_l1=abs(w) > threshold,
                interpretation=meaning,
            )
        )
    return ranked
