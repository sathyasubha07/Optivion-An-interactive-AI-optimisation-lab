from pathlib import Path

import numpy as np
import pytest

from module1.config.settings import DataConfig
from module1.data.loader import binary_and_category, load_cicids2017, make_synthetic_flows
from module1.exceptions import DatasetNotFoundError
from module1.tests.conftest import csv_with_junk


def test_binary_label_conversion():
    loaded = make_synthetic_flows(n_samples=40, n_features=4, seed=1)
    y, cats = binary_and_category(loaded.frame["Label"])
    assert set(np.unique(y).tolist()) == {0, 1}
    assert (y[cats == "BENIGN"] == 0).all()
    assert (y[cats != "BENIGN"] == 1).all()


def test_synthetic_loader_shape():
    loaded = make_synthetic_flows(n_samples=50, n_features=5, seed=2)
    assert loaded.n_rows_loaded == 50
    assert len(loaded.feature_columns) == 5


def test_missing_dataset_directory(tmp_path: Path):
    cfg = DataConfig(directory=str(tmp_path / "does_not_exist"))
    with pytest.raises(DatasetNotFoundError):
        load_cicids2017(cfg)


def test_load_csv_strips_column_spaces(tmp_path: Path):
    csv_with_junk(tmp_path / "a.csv")
    cfg = DataConfig(directory=str(tmp_path), drop_columns=["const"], min_samples=10)
    loaded = load_cicids2017(cfg)
    assert loaded.label_column.lower() == "label"
    y, cats = binary_and_category(loaded.frame[loaded.label_column])
    assert y.sum() >= 1
    assert "BENIGN" in cats
    assert "const" not in loaded.feature_columns
