from module1.services.nid_service import NIDService


def test_full_synthetic_pipeline_reproducible():
    svc = NIDService()
    a = svc.train(dataset_source="synthetic")
    b = NIDService().train(dataset_source="synthetic")
    assert a.dataset.source == "synthetic"
    assert a.evaluation.n_test > 0
    assert a.sparsity.total_features == len(a.weights)
    assert a.optimization.primal_objective is not None
    assert a.kkt.primal_feasibility >= 0
    assert a.evaluation.accuracy == b.evaluation.accuracy
    assert a.weights == b.weights
    recs = svc.predict([{"vector": [0.0] * a.dataset.n_features}])
    assert recs[0].predicted_class in {"Benign", "Attack"}
