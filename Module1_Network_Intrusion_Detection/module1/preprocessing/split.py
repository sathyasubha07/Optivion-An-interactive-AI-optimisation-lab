"""Train / validation / test split helpers. Stratified by default."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from module1.config.settings import SplitConfig
from module1.data.loader import binary_and_category
from module1.exceptions import InsufficientSamplesError
from module1.logging_setup import get_logger, log_kv

logger = get_logger(__name__)


@dataclass
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    category_train: np.ndarray
    category_val: np.ndarray
    category_test: np.ndarray


def _split_indices(indices: np.ndarray, y: np.ndarray, test_size: float, seed: int, stratify: bool):
    strat = y if stratify else None
    left, right = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=strat,
    )
    return left, right


def stratified_split(
    frame: pd.DataFrame,
    label_column: str,
    config: SplitConfig,
    seed: int,
) -> SplitFrames:
    y_bin, cats = binary_and_category(frame[label_column])
    if len(frame) < 20:
        raise InsufficientSamplesError("Need at least 20 rows to split.")
    unique, counts = np.unique(y_bin, return_counts=True)
    if unique.size < 2:
        raise InsufficientSamplesError("Both Benign and Attack must appear before splitting.")
    if counts.min() < 2:
        raise InsufficientSamplesError("Each class needs at least 2 samples for a stratified split.")

    all_idx = np.arange(len(frame))
    trainval_idx, test_idx = _split_indices(all_idx, y_bin, config.test_size, seed, config.stratify)

    if config.val_size and config.val_size > 0:
        y_tv = y_bin[trainval_idx]
        if len(np.unique(y_tv)) < 2 or np.min(np.unique(y_tv, return_counts=True)[1]) < 2:
            train_idx, val_idx = trainval_idx, np.array([], dtype=int)
        else:
            train_idx, val_idx = _split_indices(
                trainval_idx, y_tv, config.val_size, seed + 1, config.stratify
            )
    else:
        train_idx, val_idx = trainval_idx, np.array([], dtype=int)

    train = frame.iloc[train_idx].reset_index(drop=True)
    val = frame.iloc[val_idx].reset_index(drop=True) if val_idx.size else frame.iloc[[]].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    log_kv(
        logger,
        "train_val_test_split",
        train=len(train),
        val=len(val),
        test=len(test),
        stratify=config.stratify,
        seed=seed,
    )
    return SplitFrames(
        train=train,
        val=val,
        test=test,
        y_train=y_bin[train_idx],
        y_val=y_bin[val_idx] if val_idx.size else np.array([], dtype=y_bin.dtype),
        y_test=y_bin[test_idx],
        category_train=cats[train_idx],
        category_val=cats[val_idx] if val_idx.size else np.array([], dtype=cats.dtype),
        category_test=cats[test_idx],
    )
