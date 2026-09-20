"""请求/响应数据模型。"""
from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticCustomError

# 有限实数（拒绝 NaN/Inf），不确定度非负。
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegFiniteFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class UncertaintySpec(BaseModel):
    """带单位的标准不确定度；dimensionless 时可省略单位。"""

    value: NonNegFiniteFloat
    unit: str | None = Field(default=None, description="不确定度单位；缺省视为无量纲")


class InputQuantity(BaseModel):
    """一个输入量：数值、单位、标准不确定度。"""

    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="输入量名称（Python 标识符）",
    )
    value: FiniteFloat
    unit: str = Field(..., min_length=1, description="数值单位，如 m / kg / mV / rad")
    standard_uncertainty: float | UncertaintySpec = Field(
        ...,
        description="标量时与 value 同单位；对象形式 {value, unit} 可显式给出不确定度单位",
    )

    @field_validator("standard_uncertainty", mode="before")
    @classmethod
    def _validate_uncertainty_shape(cls, value: Any) -> Any:
        """按原始 JSON 类型分流，避免智能联合把标量胁迫为对象。"""
        if _is_real_number(value):
            if value < 0:
                raise PydanticCustomError(
                    "non_negative",
                    "标准不确定度必须为非负数，得到 {value}",
                    {"value": value},
                )
            return float(value)
        if isinstance(value, dict):
            # 在此校验数值，避免落入联合分支产生 'float' 之类的定位尾巴。
            num = value.get("value")
            if not _is_real_number(num) or num < 0:
                raise PydanticCustomError(
                    "non_negative",
                    "不确定度 value 必须为非负有限实数",
                )
            value = {**value, "value": float(num)}
            unit = value.get("unit")
            if unit is not None and not isinstance(unit, str):
                raise PydanticCustomError("invalid_unit", "不确定度 unit 必须为字符串")
            return value
        raise PydanticCustomError(
            "invalid_uncertainty",
            "standard_uncertainty 必须是非负数值或 {{value, unit}} 对象",
        )


class PropagationRequest(BaseModel):
    inputs: list[InputQuantity] = Field(..., min_length=1)
    formula: str = Field(..., min_length=1, description="仅含白名单运算的测量公式")
    correlation_matrix: list[list[FiniteFloat]] | None = Field(
        default=None,
        description="输入量相关系数矩阵，顺序与 inputs 一致；缺省为单位阵（互不相关）",
    )


class CorrelationTerm(BaseModel):
    """与另一输入量 j 的协方差对 i 的不确定度贡献项。"""

    input: str
    correlation: float
    covariance_contribution: float
    unit: str


class VarianceContribution(BaseModel):
    """按输入量分解的方差贡献：u_i^2 项 + 全部相关项。"""

    input: str
    variance_contribution: float = Field(..., description="u_i^2 项，(c_i u_i)^2")
    unit: str
    share: float = Field(..., description="占总方差的比例，相关项分摊后各项之和为 1")
    correlation_terms: list[CorrelationTerm]


class QuantityResult(BaseModel):
    value: float
    unit: str
    standard_uncertainty: float


class SensitivityCoefficient(BaseModel):
    input: str
    value: float
    unit: str


class PropagationResponse(BaseModel):
    result: QuantityResult
    number_of_inputs: int = Field(..., description="参与传播的输入量数量")
    correlation_matrix: list[list[float]]
    sensitivity_coefficients: list[SensitivityCoefficient]
    variance_contributions: list[VarianceContribution]
    total_variance: float
    total_variance_unit: str
    formula: str
    whitelist: list[str]
