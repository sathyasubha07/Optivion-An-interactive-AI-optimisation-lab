"""CIC-IDS2017 CSV loading and light validation.

The loader never fabricates traffic. Missing files raise DatasetNotFoundError.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from module1.config.settings import DataConfig
from module1.exceptions import DatasetFormatError, DatasetNotFoundError, InsufficientSamplesError
from module1.logging_setup import get_logger, log_kv

logger = get_logger(__name__)

BENIGN_TOKENS = {"benign", "normal", "0"}
READ_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")


@dataclass
class LoadedDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    label_column: str
    files: list[str]
    n_rows_loaded: int


def _normalize_columns(columns: list[str]) -> list[str]:
    return [str(col).replace("\n", " ").replace("\r", " ").strip() for col in columns]


def _find_label_column(columns: list[str], configured: str) -> str:
    wanted = configured.strip().lower()
    for col in columns:
        if col.strip().lower() == wanted:
            return col
    for col in columns:
        if col.strip().lower() in {"label", "class", "attack", "category"}:
            return col
    raise DatasetFormatError(
        f"Could not find a label column (configured as '{configured}'). "
        f"Columns: {columns[:30]}"
    )


def _is_benign(value: object) -> bool:
    text = str(value).strip().lower()
    return text in BENIGN_TOKENS


def binary_and_category(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Map original CIC labels to binary {0,1} while keeping the category string."""
    cleaned = series.astype(str).str.strip()
    if cleaned.str.lower().isin(["nan", "none", ""]).all():
        raise DatasetFormatError("Label column is empty.")
    binary = np.where(cleaned.map(_is_benign), 0, 1).astype(np.int32)
    if np.unique(binary).size < 2 and len(binary) > 1:
        logger.warning("Loaded labels contain only one class after BENIGN/ATTACK mapping.")
    return binary, cleaned.to_numpy()


def _read_csv(path: Path, nrows: int | None) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in READ_ENCODINGS:
        try:
            chunk = pd.read_csv(
                path,
                low_memory=False,
                nrows=nrows,
                encoding=encoding,
                skipinitialspace=True,
            )
            chunk.columns = _normalize_columns(list(chunk.columns))
            return chunk
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    raise DatasetFormatError(f"Failed to parse CSV '{path.name}': {last_error}") from last_error


def load_cicids2017(config: DataConfig) -> LoadedDataset:
    directory = config.resolved_directory()
    if not directory.exists():
        raise DatasetNotFoundError(
            f"CIC-IDS2017 directory not found: {directory}. "
            "Place the CSV files there or set data.directory in the config. "
            "See module1/data/cicids2017/README.md."
        )
    files = sorted(directory.glob(config.file_glob))
    csv_files = [path for path in files if path.is_file() and path.suffix.lower() == ".csv"]
    if not csv_files:
        raise DatasetNotFoundError(
            f"No CSV files matching '{config.file_glob}' in {directory}."
        )

    frames: list[pd.DataFrame] = []
    used_names: list[str] = []
    remaining = config.max_rows
    n_files = len(csv_files)
    for index, path in enumerate(csv_files):
        nrows = remaining
        if remaining is not None:
            leftover_files = n_files - index
            nrows = max(1, remaining // leftover_files) if leftover_files else remaining
        log_kv(logger, "loading_csv", path=path.name, nrows=nrows)
        try:
            chunk = _read_csv(path, nrows)
        except DatasetFormatError:
            raise
        if chunk.empty:
            continue
        frames.append(chunk)
        used_names.append(path.name)
        if remaining is not None:
            remaining -= len(chunk)
            if remaining <= 0:
                break

    if not frames:
        raise DatasetFormatError("CSV files were found but all were empty.")

    try:
        frame = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    except Exception as exc:  # noqa: BLE001
        raise DatasetFormatError(f"Could not concatenate CIC-IDS2017 CSVs: {exc}") from exc
    n_loaded = len(frame)
    log_kv(logger, "dataset_concatenated", rows=n_loaded, files=len(frames), columns=frame.shape[1])

    label_col = _find_label_column(list(frame.columns), config.label_column)
    drop = {name.strip() for name in config.drop_columns}
    drop_present = [c for c in frame.columns if c.strip() in drop or c in drop]
    if drop_present:
        frame = frame.drop(columns=drop_present)

    feature_columns = [c for c in frame.columns if c != label_col]
    if not feature_columns:
        raise DatasetFormatError("No feature columns remain after dropping identifiers.")
    if n_loaded < config.min_samples:
        raise InsufficientSamplesError(
            f"Loaded {n_loaded} rows; need at least {config.min_samples}."
        )
    return LoadedDataset(
        frame=frame,
        feature_columns=feature_columns,
        label_column=label_col,
        files=used_names,
        n_rows_loaded=n_loaded,
    )


def make_synthetic_flows(
    n_samples: int = 400,
    n_features: int = 12,
    attack_fraction: float = 0.35,
    seed: int = 42,
) -> LoadedDataset:
    """Linearly separable-ish flows for unit tests and API smoke checks.

    This is not CIC-IDS2017 and must not be reported as benchmark performance.
    """
    rng = np.random.default_rng(seed)
    n_attack = max(10, int(n_samples * attack_fraction))
    n_benign = n_samples - n_attack
    w_true = rng.normal(0, 1, size=n_features)
    w_true[n_features // 2 :] = 0.0
    X_b = rng.normal(0, 1, size=(n_benign, n_features))
    X_a = rng.normal(0, 1, size=(n_attack, n_features))
    X_a += 1.8 * w_true
    X = np.vstack([X_b, X_a])
    categories = np.array(["BENIGN"] * n_benign + ["DoS"] * n_attack)
    names = [f"feat_{i}" for i in range(n_features)]
    frame = pd.DataFrame(X, columns=names)
    frame["Label"] = categories
    perm = rng.permutation(n_samples)
    frame = frame.iloc[perm].reset_index(drop=True)
    return LoadedDataset(
        frame=frame,
        feature_columns=names,
        label_column="Label",
        files=["synthetic"],
        n_rows_loaded=n_samples,
    )
