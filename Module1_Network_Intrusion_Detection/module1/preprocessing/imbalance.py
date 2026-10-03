"""Class-imbalance handling applied to TRAINING data only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from module1.config.settings import ImbalanceConfig
from module1.logging_setup import get_logger, log_kv

logger = get_logger(__name__)


@dataclass
class ImbalanceResult:
    train_frame: pd.DataFrame
    y_train: np.ndarray
    category_train: np.ndarray
    sample_weights: np.ndarray
    method: str
    n_benign_before: int
    n_attack_before: int
    n_benign_after: int
    n_attack_after: int


def sklearn_balanced_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample multipliers n / (n_classes * n_c), same idea as sklearn 'balanced'."""
    y = y.astype(int)
    n = len(y)
    weights = np.ones(n, dtype=np.float64)
    for label in np.unique(y):
        n_c = int(np.sum(y == label))
        weights[y == label] = n / (2.0 * max(n_c, 1))
    return weights


def apply_imbalance(
    train: pd.DataFrame,
    y_train: np.ndarray,
    category_train: np.ndarray,
    config: ImbalanceConfig,
    seed: int,
) -> ImbalanceResult:
    n_benign_b = int(np.sum(y_train == 0))
    n_attack_b = int(np.sum(y_train == 1))
    frame = train
    y = y_train
    cats = category_train

    if config.method == "undersample":
        rng = np.random.default_rng(seed)
        idx_b = np.where(y == 0)[0]
        idx_a = np.where(y == 1)[0]
        n_a = len(idx_a)
        target_b = max(1, int(n_a * config.undersample_ratio))
        if len(idx_b) > target_b:
            keep_b = rng.choice(idx_b, size=target_b, replace=False)
            keep = np.concatenate([keep_b, idx_a])
            rng.shuffle(keep)
            frame = train.iloc[keep].reset_index(drop=True)
            y = y_train[keep]
            cats = category_train[keep]
        weights = np.ones(len(y), dtype=np.float64)
    elif config.method == "class_weight":
        weights = sklearn_balanced_weights(y)
    else:
        weights = np.ones(len(y), dtype=np.float64)

    log_kv(
        logger,
        "imbalance_applied",
        method=config.method,
        benign_before=n_benign_b,
        attack_before=n_attack_b,
        benign_after=int(np.sum(y == 0)),
        attack_after=int(np.sum(y == 1)),
    )
    return ImbalanceResult(
        train_frame=frame.reset_index(drop=True) if not isinstance(frame, pd.DataFrame) else frame,
        y_train=y,
        category_train=cats,
        sample_weights=weights,
        method=config.method,
        n_benign_before=n_benign_b,
        n_attack_before=n_attack_b,
        n_benign_after=int(np.sum(y == 0)),
        n_attack_after=int(np.sum(y == 1)),
    )
