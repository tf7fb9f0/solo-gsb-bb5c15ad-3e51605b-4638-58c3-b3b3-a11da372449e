"""Request/response schemas (Pydantic v2)."""
from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class QuantityInput(BaseModel):
    """One input quantity: nominal value, unit and standard uncertainty.

    The standard uncertainty is expressed in the same unit as the value.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$", max_length=64)
    value: FiniteFloat
    unit: str = Field(default="", max_length=64)
    uncertainty: Annotated[FiniteFloat, Field(ge=0.0)]


class CorrelationMatrix(BaseModel):
    """Input correlation matrix.

    ``order`` maps matrix rows/columns to quantity names; it must contain
    every declared quantity exactly once.
    """

    model_config = ConfigDict(extra="forbid")

    order: list[str] = Field(min_length=1, max_length=64)
    matrix: list[list[FiniteFloat]] = Field(min_length=1, max_length=64)


class PropagateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quantities: list[QuantityInput] = Field(min_length=1, max_length=64)
    formula: str = Field(min_length=1, max_length=2000)
    correlation: Optional[CorrelationMatrix] = None

    @field_validator("quantities")
    @classmethod
    def _unique_names(cls, quantities: list[QuantityInput]) -> list[QuantityInput]:
        names = [q.name for q in quantities]
        if len(set(names)) != len(names):
            raise ValueError("quantity names must be unique")
        return quantities


class Contribution(BaseModel):
    """Per-quantity contribution to the combined uncertainty.

    ``variance_contribution`` includes the correlation terms with all other
    input quantities (row sum of the contribution matrix), so the values sum
    to the combined variance; it may be negative under negative correlation.
    """

    name: str
    value: float
    unit: str
    uncertainty: float
    sensitivity: float
    sensitivity_unit: str
    standard_contribution: float
    variance_contribution: float
    variance_share: Optional[float]


class PropagateResponse(BaseModel):
    formula: str
    value: float
    unit: str
    combined_standard_uncertainty: float
    variance: float
    contributions: list[Contribution]
