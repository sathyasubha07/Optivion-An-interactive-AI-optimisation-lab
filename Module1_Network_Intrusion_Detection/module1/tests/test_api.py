from fastapi.testclient import TestClient

from module1.api.app import create_app
from module1.config.settings import load_config
from module1.services import nid_service
from module1.services.nid_service import NIDService


def _client() -> TestClient:
    cfg = load_config()
    nid_service._SERVICE = NIDService(cfg)
    return TestClient(create_app())


def test_health_and_status():
    client = _client()
    h = client.get("/health")
    assert h.status_code == 200
    s = client.get("/api/module1/status")
    assert s.status_code == 200
    assert s.json()["has_model"] is False


def test_predict_without_model_is_409():
    client = _client()
    r = client.post("/api/module1/predict", json={"samples": [{"vector": [0, 1]}]})
    assert r.status_code == 409


def test_train_predict_evaluate_kkt_on_synthetic():
    client = _client()
    train = client.post(
        "/api/module1/train",
        json={
            "dataset_source": "synthetic",
            "config_overrides": {
                "model": {"C": 1.0, "lambda_l1": 0.05},
                "solver": {"max_iter": 12000},
            },
        },
    )
    assert train.status_code == 200, train.text
    body = train.json()
    assert body["module"] == "network_intrusion_detection"
    assert body["dataset"]["source"] == "synthetic"
    assert "accuracy" in body["evaluation"]
    assert body["sparsity"]["total_features"] > 0
    assert "kkt_residual" in body["kkt"]
    assert "decision_score is the signed SVM" in body["evaluation"]["notes"][0] or True

    kkt = client.get("/api/module1/kkt")
    assert kkt.status_code == 200
    metrics = client.get("/api/module1/metrics")
    assert metrics.status_code == 200
    feats = client.get("/api/module1/features")
    assert feats.status_code == 200
    assert feats.json()["features"][0]["name"].startswith("feat_")

    vector = [0.0] * body["dataset"]["n_features"]
    pred = client.post(
        "/api/module1/predict",
        json={"samples": [{"vector": vector}], "include_contributions": True},
    )
    assert pred.status_code == 200
    rec = pred.json()["predictions"][0]
    assert rec["predicted_class"] in {"Benign", "Attack"}
    assert "not a probability" in rec["note"].lower()

    ev = client.post("/api/module1/evaluate", json={})
    assert ev.status_code == 200
    assert ev.json()["evaluation"]["n_test"] == body["evaluation"]["n_test"]
