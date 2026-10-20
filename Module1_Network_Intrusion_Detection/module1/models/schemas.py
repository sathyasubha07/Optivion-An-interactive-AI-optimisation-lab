"""Pydantic contracts consumed by the API and (later) the OPTIVION UI."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    name: str
    source: Literal["cicids2017", "synthetic"]
    evaluation_protocol: Literal["development_sample", "configured_dataset"]
    n_rows_loaded: int
    n_rows_after_clean: int
    train_samples: int
    validation_samples: int = 0
    test_samples: int
    n_features: int
    n_benign_loaded: int
    n_attack_loaded: int
    n_benign_train_before_balance: int
    n_attack_train_before_balance: int
    n_benign_train_after_balance: int
    n_attack_train_after_balance: int
    n_benign_test: int
    n_attack_test: int
    files: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ModelSummary(BaseModel):
    type: str = "soft_margin_linear_svm_l1"
    formulation: str
    C: float
    lambda_l1: float
    decision_threshold: float
    sparsity_threshold: float
    random_seed: int
    class_weight_applied: bool
    solver_name: str = "osqp"
    hyperparameters_tuned: bool = False
    threshold_selected_on_validation: bool = False
    validation_samples: int = 0
    selection_score: float | None = None


class OptimizationReport(BaseModel):
    status: str
    status_val: int | None = None
    converged: bool
    objective_value: float | None = None
    primal_objective: float | None = None
    iterations: int | None = None
    training_time_seconds: float
    polish_status: str | None = None
    primal_residual: float | None = None
    dual_residual: float | None = None
    run_time: float | None = None
    notes: list[str] = Field(default_factory=list)


class KKTReport(BaseModel):
    primal_feasibility: float
    dual_feasibility: float | None
    complementary_slackness: float | None
    stationarity_residual: float | None
    max_constraint_violation: float
    kkt_residual: float
    within_tolerance: bool
    tolerance: float
    alpha_box_violation: float | None
    max_margin_violation: float
    max_nonnegativity_violation: float
    stationarity_b: float | None
    limitations: list[str] = Field(default_factory=list)


class SparsityReport(BaseModel):
    total_features: int
    nonzero_features: int
    zero_features: int
    sparsity_percentage: float
    threshold: float
    l1_norm: float
    l2_norm: float


class FeatureWeight(BaseModel):
    rank: int
    name: str
    weight: float
    abs_weight: float
    retained_by_l1: bool
    interpretation: str


class SupportVectorReport(BaseModel):
    count: int
    fraction_of_training: float
    indices: list[int] = Field(default_factory=list)
    alpha_min: float | None = None
    alpha_max: float | None = None
    mean_abs_margin_on_sv: float | None = None
    notes: list[str] = Field(default_factory=list)


class ConfusionCounts(BaseModel):
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


class EvaluationReport(BaseModel):
    n_test: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    attack_precision: float
    attack_recall: float
    attack_f1: float
    benign_precision: float
    benign_recall: float
    false_positive_rate: float
    false_negative_rate: float
    confusion_matrix: ConfusionCounts
    inference_time_seconds: float
    inference_time_per_sample_seconds: float
    notes: list[str] = Field(default_factory=list)


class PredictionRecord(BaseModel):
    predicted_label: int
    predicted_class: Literal["Benign", "Attack"]
    decision_score: float
    margin: float
    on_margin: bool
    feature_contributions: list[dict[str, Any]] | None = None
    note: str = (
        "decision_score is the signed SVM value w^T x + b. "
        "It is not a probability."
    )


class Module1Result(BaseModel):
    module: str = "network_intrusion_detection"
    dataset: DatasetSummary
    model: ModelSummary
    optimization: OptimizationReport
    kkt: KKTReport
    sparsity: SparsityReport
    support_vectors: SupportVectorReport
    evaluation: EvaluationReport
    features: list[FeatureWeight]
    weights: list[float]
    bias: float
    software: dict[str, str] = Field(default_factory=dict)


class TrainRequest(BaseModel):
    dataset_source: Literal["cicids2017", "synthetic"] = "cicids2017"
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    samples: list[dict[str, Any]]
    include_contributions: bool = True
    top_k_contributions: int = 8


class EvaluateRequest(BaseModel):
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class ExperimentRequest(BaseModel):
    dataset_source: Literal["cicids2017", "synthetic"] = "cicids2017"
    compare_baseline: bool = True
    C_values: list[float] | None = None
    lambda_values: list[float] | None = None
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class StatusResponse(BaseModel):
    module: str = "network_intrusion_detection"
    state: Literal["idle", "training", "ready", "error"]
    message: str
    has_model: bool
    last_train_protocol: str | None = None
