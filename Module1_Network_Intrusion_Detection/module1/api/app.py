"""FastAPI application for Module 1. JSON only — no HTML UI."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from module1 import MODULE_NAME, __version__
from module1.api.routes import router
from module1.exceptions import Module1Error
from module1.logging_setup import configure_logging, get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="OPTIVION Module 1 — Network Intrusion Detection",
        description=(
            "Backend API for the SVM+L1 intrusion detector. "
            "The OPTIVION UI does not exist yet; this interface is the integration contract."
        ),
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.include_router(router, prefix="/api/module1", tags=["module1"])

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "module": MODULE_NAME}

    @app.exception_handler(Module1Error)
    async def module_error(_: Request, exc: Module1Error) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": exc.code, "message": exc.message},
        )

    return app


app = create_app()
