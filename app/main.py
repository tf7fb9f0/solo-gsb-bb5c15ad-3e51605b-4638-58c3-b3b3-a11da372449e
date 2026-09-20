"""FastAPI 应用入口。"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from .errors import UnprocessableError, request_validation_handler, unprocessable_handler
from .formula import WHITELIST_DISPLAY
from .propagation import propagate
from .schemas import PropagationRequest, PropagationResponse

app = FastAPI(
    title="校准计算不确定度传播 API",
    description=(
        "提交带数值、单位、标准不确定度的输入量、相关系数矩阵和白名单测量公式，"
        "服务端进行量纲校验，并用符号偏导与协方差矩阵计算合成标准不确定度（GUM 一阶）。"
    ),
    version="1.0.0",
)

app.add_exception_handler(UnprocessableError, unprocessable_handler)
app.add_exception_handler(RequestValidationError, request_validation_handler)


@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/v1/whitelist", tags=["uncertainty"])
def whitelist() -> dict:
    """返回公式允许使用的运算与函数。"""
    return {"whitelist": WHITELIST_DISPLAY}


_EXAMPLE = {
    "inputs": [
        {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
        {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.02},
    ],
    "formula": "U*I",
    "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
}


@app.post(
    "/api/v1/propagate",
    response_model=PropagationResponse,
    tags=["uncertainty"],
    responses={422: {"description": "相关矩阵非法或单位/量纲不兼容等"}},
    openapi_extra={
        "examples": {"欧姆定律 P=U*I": {"value": _EXAMPLE}},
    },
)
def calculate(req: PropagationRequest) -> dict:
    """执行一次不确定度传播计算。"""
    return propagate(req)
