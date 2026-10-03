import numpy as np

from module1.config.settings import ImbalanceConfig, PreprocessingConfig, SplitConfig
from module1.data.loader import make_synthetic_flows
from module1.preprocessing.imbalance import apply_imbalance
from module1.preprocessing.pipeline import clean_frame, fit_preprocessor
from module1.preprocessing.split import stratified_split


def test_clean_and_split_no_overlap():
    loaded = make_synthetic_flows(n_samples=120, n_features=6, seed=3)
    cleaned, notes = clean_frame(loaded, drop_duplicates=True)
    assert len(cleaned) >= 50
    split = stratified_split(cleaned, loaded.label_column, SplitConfig(test_size=0.25), seed=3)
    train_t = set(map(tuple, split.train.to_numpy()))
    test_t = set(map(tuple, split.test.to_numpy()))
    assert train_t.isdisjoint(test_t)
    assert split.y_train.size == len(split.train)
    assert 0 in split.y_train and 1 in split.y_train
    assert 0 in split.y_test and 1 in split.y_test
    assert notes


def test_scaler_fit_on_train_only():
    loaded = make_synthetic_flows(n_samples=80, n_features=5, seed=4)
    cleaned, _ = clean_frame(loaded, drop_duplicates=True)
    split = stratified_split(cleaned, loaded.label_column, SplitConfig(), seed=4)
    names = [c for c in split.train.columns if c != loaded.label_column]
    pre = fit_preprocessor(split.train, names, PreprocessingConfig())
    X_train = pre.transform(split.train)
    X_test = pre.transform(split.test)
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_test).all()
    # Training features should be approximately standardized.
    assert np.max(np.abs(X_train.mean(axis=0))) < 1e-6
    assert np.max(np.abs(X_train.std(axis=0) - 1.0)) < 0.15


def test_undersample_train_only():
    loaded = make_synthetic_flows(n_samples=200, n_features=4, attack_fraction=0.2, seed=5)
    cleaned, _ = clean_frame(loaded, drop_duplicates=False)
    split = stratified_split(cleaned, loaded.label_column, SplitConfig(), seed=5)
    n_test_before = len(split.test)
    imb = apply_imbalance(
        split.train,
        split.y_train,
        split.category_train,
        ImbalanceConfig(method="undersample", undersample_ratio=1.0),
        seed=5,
    )
    assert imb.n_benign_after <= imb.n_benign_before
    assert len(split.test) == n_test_before
    assert imb.n_attack_after == imb.n_attack_before
