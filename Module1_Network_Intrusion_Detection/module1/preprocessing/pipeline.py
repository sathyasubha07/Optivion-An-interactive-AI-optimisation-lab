"""Leakage-safe preprocessing for CIC-IDS2017 tabular flows."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from module1.config.settings import PreprocessingConfig
from module1.data.loader import LoadedDataset, binary_and_category
from module1.exceptions import DatasetFormatError
from module1.logging_setup import get_logger, log_kv

logger = get_logger(__name__)


@dataclass
class FittedPreprocessor:
    feature_names: list[str]
    source_feature_names: list[str]
    keep_mask: np.ndarray
    imputer: SimpleImputer | None
    scaler: object | None
    fill_inf: bool
    notes: list[str] = field(default_factory=list)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = _to_numeric_matrix(frame, self.source_feature_names, fill_inf=self.fill_inf)
        if self.imputer is not None:
            numeric = self.imputer.transform(numeric)
        numeric = numeric[:, self.keep_mask]
        if self.scaler is not None:
            numeric = self.scaler.transform(numeric)
        if not np.isfinite(numeric).all():
            raise DatasetFormatError("Non-finite values remain after preprocessing.")
        return numeric.astype(np.float64)


@dataclass
class PreparedTable:
    X: np.ndarray
    y_binary: np.ndarray
    attack_category: np.ndarray
    feature_names: list[str]
    n_rows_after_clean: int
    notes: list[str]


def _to_numeric_matrix(frame: pd.DataFrame, columns: list[str], fill_inf: bool) -> np.ndarray:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise DatasetFormatError(f"Missing feature columns: {missing[:10]}")
    subset = frame.loc[:, columns]
    coerced = subset.apply(pd.to_numeric, errors="coerce")
    values = np.array(coerced.to_numpy(dtype=np.float64), copy=True)
    if fill_inf:
        values[~np.isfinite(values)] = np.nan
    return values


def clean_frame(loaded: LoadedDataset, drop_duplicates: bool) -> tuple[pd.DataFrame, list[str]]:
    frame = loaded.frame.copy()
    notes: list[str] = []
    n0 = len(frame)
    frame = frame.loc[:, loaded.feature_columns + [loaded.label_column]]
    numeric_probe = frame.loc[:, loaded.feature_columns].apply(pd.to_numeric, errors="coerce")
    unusable = []
    for col in loaded.feature_columns:
        series = numeric_probe[col]
        if series.notna().sum() == 0:
            unusable.append(col)
    if unusable:
        frame = frame.drop(columns=unusable)
        notes.append(f"Dropped {len(unusable)} all-missing columns before the split.")
        log_kv(logger, "dropped_unusable_columns", count=len(unusable))
    if drop_duplicates:
        before = len(frame)
        frame = frame.drop_duplicates()
        notes.append(f"Removed {before - len(frame)} duplicate rows.")
    notes.append(f"Cleaned rows: {n0} -> {len(frame)}.")
    return frame.reset_index(drop=True), notes


def extract_xy(frame: pd.DataFrame, label_column: str) -> PreparedTable:
    y_bin, categories = binary_and_category(frame[label_column])
    feature_names = [c for c in frame.columns if c != label_column]
    if not feature_names:
        raise DatasetFormatError("No usable numeric features after cleaning.")
    notes = [
        "Categorical/non-numeric feature cells were coerced with pandas.to_numeric(errors='coerce') "
        "and imputed from training statistics only."
    ]
    return PreparedTable(
        X=frame,  # type: ignore[arg-type]
        y_binary=y_bin,
        attack_category=categories,
        feature_names=feature_names,
        n_rows_after_clean=len(frame),
        notes=notes,
    )


def fit_preprocessor(
    train_frame: pd.DataFrame,
    feature_names: list[str],
    config: PreprocessingConfig,
) -> FittedPreprocessor:
    matrix = _to_numeric_matrix(train_frame, feature_names, fill_inf=config.fill_inf)
    imputer = SimpleImputer(strategy=config.impute_strategy)
    imputed = imputer.fit_transform(matrix)
    keep = np.std(imputed, axis=0) > 1e-12
    if not bool(np.any(keep)):
        raise DatasetFormatError("All training features are constant after imputation.")
    kept_names = [name for name, flag in zip(feature_names, keep.tolist()) if flag]
    imputed_kept = imputed[:, keep]
    dropped = int(np.size(keep) - int(np.sum(keep)))
    scaler: object | None
    if config.scaler == "standard":
        scaler = StandardScaler()
        scaler.fit(imputed_kept)
    elif config.scaler == "minmax":
        scaler = MinMaxScaler()
        scaler.fit(imputed_kept)
    else:
        scaler = None
    log_kv(logger, "preprocessor_fitted", features=len(kept_names), scaler=config.scaler, dropped_constant=dropped)
    notes = [
        "Imputer and scaler were fit on the training split only.",
        "The test split is transformed with stored statistics (no leakage).",
        "Constant columns were detected on the training split only.",
    ]
    if dropped:
        notes.append(f"Dropped {dropped} train-constant columns.")
    return FittedPreprocessor(
        feature_names=kept_names,
        source_feature_names=feature_names,
        keep_mask=keep,
        imputer=imputer,
        scaler=scaler,
        fill_inf=config.fill_inf,
        notes=notes,
    )
