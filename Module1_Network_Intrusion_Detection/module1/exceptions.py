"""Module 1 exception types with API-friendly error codes."""


class Module1Error(Exception):
    """Base error for the intrusion-detection backend."""

    def __init__(self, message: str, code: str = "module1_error") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class DatasetNotFoundError(Module1Error):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="dataset_not_found")


class DatasetFormatError(Module1Error):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="dataset_format_error")


class InsufficientSamplesError(Module1Error):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="insufficient_samples")


class InvalidHyperparameterError(Module1Error):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="invalid_hyperparameter")


class OptimizationError(Module1Error):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="optimization_failure")


class ModelNotTrainedError(Module1Error):
    def __init__(self, message: str = "No trained model is available. Train Module 1 first.") -> None:
        super().__init__(message, code="model_not_trained")


class PredictionInputError(Module1Error):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="prediction_input_error")


class ConfigurationError(Module1Error):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="configuration_error")
