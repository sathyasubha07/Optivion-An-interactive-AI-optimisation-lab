"""HTTP contract for future OPTIVION UI consumption."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from module1.exceptions import ModelNotTrainedError, Module1Error
from module1.models.schemas import (
    EvaluateRequest,
    ExperimentRequest,
    PredictRequest,
    StatusResponse,
    TrainRequest,
)
from module1.services.nid_service import get_service

router = APIRouter()


@router.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    svc = get_service()
    protocol = None
    if svc.state is not None:
        protocol = svc.state.result.dataset.evaluation_protocol
    return StatusResponse(
        state=svc.status,
        message=svc.status_message,
        has_model=svc.state is not None,
        last_train_protocol=protocol,
    )


@router.post("/train")
def train(body: TrainRequest):
    svc = get_service()
    result = svc.train(dataset_source=body.dataset_source, config_overrides=body.config_overrides)
    return result.model_dump()


@router.post("/predict")
def predict(body: PredictRequest):
    svc = get_service()
    try:
        records = svc.predict(
            body.samples,
            include_contributions=body.include_contributions,
            top_k=body.top_k_contributions,
        )
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=409, detail={"error": exc.code, "message": exc.message}) from exc
    except Module1Error as exc:
        raise HTTPException(status_code=400, detail={"error": exc.code, "message": exc.message}) from exc
    return {"module": "network_intrusion_detection", "predictions": [r.model_dump() for r in records]}


@router.post("/evaluate")
def evaluate(body: EvaluateRequest | None = None):
    svc = get_service()
    _ = body
    try:
        return svc.evaluate_current().model_dump()
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=409, detail={"error": exc.code, "message": exc.message}) from exc


@router.post("/experiment")
def experiment(body: ExperimentRequest):
    svc = get_service()
    return svc.experiment(
        dataset_source=body.dataset_source,
        compare_baseline=body.compare_baseline,
        C_values=body.C_values,
        lambda_values=body.lambda_values,
        config_overrides=body.config_overrides,
    )


@router.get("/model-info")
def model_info():
    svc = get_service()
    try:
        result = svc.require_state().result
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=409, detail={"error": exc.code, "message": exc.message}) from exc
    return {
        "dataset": result.dataset.model_dump(),
        "model": result.model.model_dump(),
        "optimization": result.optimization.model_dump(),
        "sparsity": result.sparsity.model_dump(),
        "support_vectors": {
            "count": result.support_vectors.count,
            "fraction_of_training": result.support_vectors.fraction_of_training,
        },
        "bias": result.bias,
        "software": result.software,
    }


@router.get("/features")
def features():
    svc = get_service()
    try:
        result = svc.require_state().result
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=409, detail={"error": exc.code, "message": exc.message}) from exc
    return {"features": [f.model_dump() for f in result.features], "sparsity": result.sparsity.model_dump()}


@router.get("/kkt")
def kkt():
    svc = get_service()
    try:
        return svc.require_state().result.kkt.model_dump()
    except ModelNotTrainedError as orig:
        raise HTTPException(status_code=409, detail={"error": orig.code, "message": orig.message}) from orig


@router.get("/metrics")
def metrics():
    svc = get_service()
    try:
        result = svc.require_state().result
    except ModelNotTrainedError as orig:
        raise HTTPException(status_code=409, detail={"error": orig.code, "message": orig.message}) from orig
    return {
        "evaluation": result.evaluation.model_dump(),
        "optimization": result.optimization.model_dump(),
        "sparsity": result.sparsity.model_dump(),
        "evaluation_protocol": result.dataset.evaluation_protocol,
        "dataset_source": result.dataset.source,
    }
