"""YAML-backed configuration for Module 1.

Absolute user-specific paths are never required. Dataset location is a
repository-relative path unless the caller overrides it.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from module1.exceptions import ConfigurationError, InvalidHyperparameterError


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default.yaml"


class DataConfig(BaseModel):
    directory: str = "module1/data/cicids2017"
    file_glob: str = "*.csv"
    label_column: str = "Label"
    drop_columns: list[str] = Field(default_factory=list)
    drop_duplicates: bool = True
    max_rows: int | None = None
    use_development_sample: bool = True
    development_sample_size: int = 8000
    min_samples: int = 50

    def resolved_directory(self) -> Path:
        path = Path(self.directory)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists() and path.name == "cicids2017" and    path.parent.name == "data":
            alt = REPO_ROOT / "module1" / "data" / "cicids2017"
            if alt.exists():
                return alt
        return path


class SplitConfig(BaseModel):
    test_size: float = 0.2
    val_size: float = 0.2
    stratify: bool = True

    @field_validator("test_size")
    @classmethod
    def _test_size_range(cls, value: float) -> float:
        if not 0.05 <= value <= 0.5:
            raise InvalidHyperparameterError("split.test_size must be in [0.05, 0.5]")
        return value

    @field_validator("val_size")
    @classmethod
    def _val_size_range(cls, value: float) -> float:
        if not 0.0 <= value <= 0.5:
            raise InvalidHyperparameterError("split.val_size must be in [0.0, 0.5]")
        return value


class PreprocessingConfig(BaseModel):
    fill_inf: bool = True
    impute_strategy: str = "median"
    scaler: str = "standard"

    @field_validator("impute_strategy")
    @classmethod
    def _impute(cls, value: str) -> str:
        allowed = {"median", "mean"}
        if value not in allowed:
            raise InvalidHyperparameterError(f"impute_strategy must be one of {allowed}")
        return value

    @field_validator("scaler")
    @classmethod
    def _scaler(cls, value: str) -> str:
        allowed = {"standard", "minmax", "none"}
        if value not in allowed:
            raise InvalidHyperparameterError(f"scaler must be one of {allowed}")
        return value


class ImbalanceConfig(BaseModel):
    method: str = "class_weight"
    undersample_ratio: float = 1.0

    @field_validator("method")
    @classmethod
    def _method(cls, value: str) -> str:
        allowed = {"class_weight", "undersample", "none"}
        if value not in allowed:
            raise InvalidHyperparameterError(f"imbalance.method must be one of {allowed}")
        return value


class TuningConfig(BaseModel):
    enabled: bool = True
    C_values: list[float] = Field(default_factory=lambda: [0.3, 1.0, 3.0])
    lambda_values: list[float] = Field(default_factory=lambda: [0.01, 0.05, 0.2])
    select_threshold: bool = True
    fpr_penalty: float = 0.25

    @field_validator("fpr_penalty")
    @classmethod
    def _pen(cls, value: float) -> float:
        if value < 0:
            raise InvalidHyperparameterError("tuning.fpr_penalty must be >= 0")
        return value

    @field_validator("C_values")
    @classmethod
    def _cs(cls, value: list[float]) -> list[float]:
        if not value or any(c <= 0 for c in value):
            raise InvalidHyperparameterError("tuning.C_values must be positive")
        return value

    @field_validator("lambda_values")
    @classmethod
    def _lams(cls, value: list[float]) -> list[float]:
        if not value or any(lam < 0 for lam in value):
            raise InvalidHyperparameterError("tuning.lambda_values must be >= 0")
        return value


class ModelConfig(BaseModel):
    C: float = 1.0
    lambda_l1: float = 0.05
    sparsity_threshold: float = 1e-6
    decision_threshold: float = 0.0

    @field_validator("C")
    @classmethod
    def _c(cls, value: float) -> float:
        if value <= 0:
            raise InvalidHyperparameterError("model.C must be > 0")
        return value

    @field_validator("lambda_l1")
    @classmethod
    def _lam(cls, value: float) -> float:
        if value < 0:
            raise InvalidHyperparameterError("model.lambda_l1 must be >= 0")
        return value

    @field_validator("sparsity_threshold")
    @classmethod
    def _thr(cls, value: float) -> float:
        if value < 0:
            raise InvalidHyperparameterError("sparsity_threshold must be >= 0")
        return value


class SolverConfig(BaseModel):
    max_iter: int = 20000
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4
    polish: bool = True
    verbose: bool = False
    kkt_tolerance: float = 1e-3
    support_vector_alpha_tol: float = 1e-5


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Module1Config(BaseModel):
    module_name: str = "network_intrusion_detection"
    random_seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    imbalance: ImbalanceConfig = Field(default_factory=ImbalanceConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def merged(self, overrides: dict[str, Any] | None) -> "Module1Config":
        if not overrides:
            return self.model_copy(deep=True)
        payload = deep_merge(self.model_dump(), overrides)
        return Module1Config.model_validate(payload)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path | None = None) -> Module1Config:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        candidate = Path.cwd() / config_path
        config_path = candidate if candidate.exists() else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    try:
        return Module1Config.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 — surface YAML/validation issues uniformly
        if isinstance(exc, InvalidHyperparameterError):
            raise
        raise ConfigurationError(f"Invalid configuration: {exc}") from exc
