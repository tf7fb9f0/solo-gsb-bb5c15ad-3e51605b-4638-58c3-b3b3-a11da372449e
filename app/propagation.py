"""不确定度传播主流程：符号偏导 + 协方差矩阵（GUM 一阶近似）。

对测量模型 y = f(x1, ..., xn)，合成标准不确定度：

    u_c^2 = Σ_i c_i^2 u(x_i)^2
          + 2 Σ_{i<j} c_i c_j u(x_i) u(x_j) r(x_i, x_j)

其中 c_i = ∂f/∂x_i 在输入估计值处取值。
"""
from __future__ import annotations

import math

import sympy as sp

from .correlation import validate_correlation_matrix
from .errors import UnprocessableError
from .formula import RESERVED_NAMES, dimension_of, evaluate, parse_formula
from .units import (
    Quantity,
    convert_magnitude,
    parse_unit,
    preferred_unit,
    same_dimensionality,
    unit_str,
)


def _prepare_inputs(req) -> list[dict]:
    """解析每个输入量的单位与不确定度，返回内部表示列表。"""
    seen: set[str] = set()
    prepared: list[dict] = []

    for index, item in enumerate(req.inputs):
        loc = ["inputs", index]
        if item.name in seen:
            raise UnprocessableError(
                f"输入量名称 {item.name!r} 重复", loc + ["name"], code="duplicate_name"
            )
        if item.name in RESERVED_NAMES:
            raise UnprocessableError(
                f"输入量名称 {item.name!r} 是白名单保留名称，请更换",
                loc + ["name"],
                code="reserved_name",
            )
        seen.add(item.name)

        unit = parse_unit(item.unit, loc + ["unit"])
        q = Quantity(float(item.value), unit)

        su = item.standard_uncertainty
        if isinstance(su, (int, float)):
            u_mag, u_unit = float(su), unit
        else:
            if su.unit is None:
                u_mag, u_unit = float(su.value), unit
            else:
                u_unit = parse_unit(su.unit, loc + ["standard_uncertainty", "unit"])
                if not same_dimensionality(u_unit, unit):
                    raise UnprocessableError(
                        f"不确定度单位 {u_unit:~} 与输入量单位 {unit:~} 量纲不一致",
                        loc + ["standard_uncertainty", "unit"],
                        code="incompatible_unit",
                    )
                u_mag = convert_magnitude(
                    Quantity(float(su.value), u_unit),
                    unit,
                    loc + ["standard_uncertainty", "value"],
                )

        prepared.append(
            {
                "name": item.name,
                "symbol": sp.Symbol(item.name),
                "value": q,
                "magnitude": float(item.value),
                "unit": unit,
                "u_magnitude": u_mag,  # 以输入量自身单位表示
            }
        )

    return prepared


def propagate(req) -> dict:
    """执行完整传播计算，返回可直接序列化的结果字典。"""
    inputs = _prepare_inputs(req)
    n = len(inputs)

    corr = validate_correlation_matrix(req.correlation_matrix, n)

    names = {p["name"] for p in inputs}
    expr = parse_formula(req.formula, names)

    # 量纲静态校验。
    symbol_dims = {p["symbol"]: p["unit"].dimensionality for p in inputs}
    dimension_of(expr, symbol_dims)

    quantities = {p["symbol"]: p["value"] for p in inputs}

    # 结果值（带单位求值），并把 A·V 这类单位乘积规约为常用导出单位（W）。
    result_q = evaluate(expr, quantities)
    if not math.isfinite(float(result_q.magnitude)):
        raise UnprocessableError(
            "公式求值结果不是有限实数，请检查输入与公式",
            ["formula"],
            code="evaluation_error",
        )
    display_unit = preferred_unit(result_q.units)
    result_value = convert_magnitude(result_q, display_unit, ["formula"])
    result_q = Quantity(result_value, display_unit)

    # 符号偏导并在输入估计值处求值。
    sensitivities: list[dict] = []
    cu_mags: list[float] = []  # c_i u_i，统一用结果单位表示的数值
    for p in inputs:
        derivative = sp.diff(expr, p["symbol"])
        deriv_q = evaluate(derivative, quantities)
        # c_i 单位 = 结果单位 / 输入单位（量纲已由静态校验保证）。
        deriv_unit = result_q.units / p["unit"]
        zero_magnitude = float(deriv_q.magnitude) == 0.0
        if zero_magnitude:
            # 偏导恒为零（输入量未出现在公式中）：零可带任意单位。
            c_mag = 0.0
        else:
            c_mag = convert_magnitude(deriv_q, deriv_unit, ["formula"])
        if not math.isfinite(c_mag):
            raise UnprocessableError(
                f"对 {p['name']} 的偏导在给定输入处不是有限实数，"
                "无法进行一阶不确定度传播",
                ["formula"],
                code="evaluation_error",
            )
        cu_mags.append(c_mag * p["u_magnitude"])
        sensitivities.append(
            {
                "input": p["name"],
                "value": c_mag,
                "unit": unit_str(deriv_unit),
            }
        )

    # 协方差矩阵传播：C_y = c_u^T R c_u（数值以结果单位计）。
    cu = cu_mags
    variance_matrix = [[cu[i] * corr[i, j] * cu[j] for j in range(n)] for i in range(n)]
    total_variance = sum(sum(row) for row in variance_matrix)
    if total_variance < 0 and total_variance > -1e-12:
        total_variance = 0.0
    if total_variance < 0:
        raise UnprocessableError(
            f"传播后方差为负（{total_variance:.6g}），相关结构数值不自洽",
            ["correlation_matrix"],
            code="not_positive_semidefinite",
        )
    uc = math.sqrt(total_variance)

    # 按输入量分解方差贡献：
    #   对角项 u_i^2 = (c_i u_i)^2；
    #   交叉项 2 c_i c_j u_i u_j r_ij 在 i、j 上各分摊一半，
    #   因而各输入量贡献之和恰为总方差。
    contributions: list[dict] = []
    shares = [cu[i] * cu[i] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            pair = 2.0 * variance_matrix[i][j]
            if abs(pair) < 1e-300:
                pair = 0.0
            shares[i] += pair / 2.0
            shares[j] += pair / 2.0

    variance_display_unit = unit_str(display_unit ** 2)
    for i, p in enumerate(inputs):
        corr_terms = []
        for j in range(n):
            if i == j:
                continue
            pair = 2.0 * variance_matrix[i][j]
            if abs(pair) < 1e-300:
                pair = 0.0
            corr_terms.append(
                {
                    "input": inputs[j]["name"],
                    "correlation": float(corr[i, j]),
                    "covariance_contribution": pair,
                    "unit": variance_display_unit,
                }
            )
        share = shares[i] / total_variance if total_variance > 0 else 0.0
        contributions.append(
            {
                "input": p["name"],
                "variance_contribution": shares[i],
                "unit": variance_display_unit,
                "share": share,
                "correlation_terms": corr_terms,
            }
        )

    return {
        "result": {
            "value": float(result_q.magnitude),
            "unit": unit_str(result_q.units),
            "standard_uncertainty": uc,
        },
        "number_of_inputs": n,
        "correlation_matrix": [[float(x) for x in row] for row in corr],
        "sensitivity_coefficients": sensitivities,
        "variance_contributions": contributions,
        "total_variance": total_variance,
        "total_variance_unit": variance_display_unit,
        "formula": req.formula,
        "whitelist": [
            "+", "-", "*", "/", "** (^)",
            "sqrt", "exp", "log", "ln", "log10", "log2",
            "sin", "cos", "tan", "asin", "acos", "atan", "abs",
            "pi", "E",
        ],
    }
