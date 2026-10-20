import pytest

from module1.config.settings import DataConfig, load_config
from module1.exceptions import DatasetNotFoundError, InvalidHyperparameterError
from module1.services.nid_service import NIDService


def test_invalid_C():
    with pytest.raises(InvalidHyperparameterError):
        load_config().__class__.model_validate({"model": {"C": -1}})


def test_train_cicids_missing(tmp_path):
    cfg = load_config()
    cfg.data = DataConfig(directory=str(tmp_path / "empty"), min_samples=10)
    svc = NIDService(cfg)
    with pytest.raises(DatasetNotFoundError):
        svc.train(dataset_source="cicids2017")
