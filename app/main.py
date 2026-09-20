"""Calibration uncertainty propagation API (FastAPI + SymPy + Pint)."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .engine import QuantitySpec, propagate
from .errors import DomainError
from .schemas import PropagateRequest, PropagateResponse

app = FastAPI(
    title="Calibration Uncertainty Propagation API",
    version="1.0.0",
    summary="GUM law of propagation of uncertainty with units and correlations",
)


@app.exception_handler(DomainError)
async def domain_error_handler(_: Request, exc: DomainError) -> JSONResponse:
    """Domain validation failures are reported as 422 with a precise location."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {
                    "loc": ["body", *exc.loc],
                    "msg": exc.msg,
                    "type": "value_error.domain",
                }
            ]
        },
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/propagate", response_model=PropagateResponse)
def propagate_endpoint(request: PropagateRequest) -> dict:
    specs = [
        QuantitySpec(name=q.name, value=q.value, unit=q.unit, uncertainty=q.uncertainty)
        for q in request.quantities
    ]
    return propagate(specs, request.formula, request.correlation)
