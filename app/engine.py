"""Uncertainty propagation engine.

Implements the GUM law of propagation of uncertainty:

    u_c^2(y) = sum_i sum_j  c_i * c_j * cov(x_i, x_j)
             = t^T R t        with t_i = c_i * u(x_i)

where c_i = df/dx_i is the sensitivity coefficient (symbolic partial
derivative via SymPy, evaluated with Pint quantities so units are checked
and carried through), u(x_i) the standard uncertainty and R the input
correlation matrix.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass

import numpy as np
import pint
import sympy as sp

from .errors import DomainError

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# ---------------------------------------------------------------------------
# Formula parsing (Python AST -> SymPy, whitelist only)
# ---------------------------------------------------------------------------

_MAX_NODES = 400            # AST node budget per formula
_MAX_LITERAL = 1e12         # magnitude limit for numeric literals
_MAX_CONST_EXPONENT = 100   # limit for constant exponents

# Whitelisted single-argument functions (dimensionless in, dimensionless out,
# except abs() which preserves units).
_FUNCTIONS = {
    "sqrt": sp.sqrt,
    "exp": sp.exp,
    "ln": sp.log,
    "log": sp.log,
    "log10": lambda x: sp.log(x, 10),
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "abs": sp.Abs,
}
_CONSTANTS = {"pi": sp.pi, "e": sp.E}
_BINOP_NAMES = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}


def parse_formula(formula: str, symbols: dict[str, sp.Symbol]) -> sp.Expr:
    """Parse a measurement formula into a SymPy expression.

    Only + - * / **, parentheses, numeric literals, declared quantity
    symbols, the constants pi/e and whitelisted functions are accepted.
    """
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise DomainError(["formula"], f"invalid syntax: {exc.msg}") from exc
    return _convert(tree.body, symbols, [0])


def _convert(node: ast.AST, symbols: dict[str, sp.Symbol], counter: list[int]) -> sp.Expr:
    counter[0] += 1
    if counter[0] > _MAX_NODES:
        raise DomainError(["formula"], f"formula is too complex (limit of {_MAX_NODES} nodes)")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINOP_NAMES:
            raise DomainError(
                ["formula"],
                f"operator '{type(node.op).__name__}' is not allowed; use one of + - * / **",
            )
        left = _convert(node.left, symbols, counter)
        right = _convert(node.right, symbols, counter)
        if op_type is ast.Pow:
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)) \
                    and not isinstance(node.right.value, bool):
                if abs(node.right.value) > _MAX_CONST_EXPONENT:
                    raise DomainError(
                        ["formula"],
                        f"constant exponent exceeds the allowed limit of +/-{_MAX_CONST_EXPONENT}",
                    )
            return left ** right
        if op_type is ast.Add:
            return left + right
        if op_type is ast.Sub:
            return left - right
        if op_type is ast.Mult:
            return left * right
        return left / right

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_convert(node.operand, symbols, counter)
        if isinstance(node.op, ast.UAdd):
            return _convert(node.operand, symbols, counter)
        raise DomainError(["formula"], "unary operator is not allowed")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise DomainError(["formula"], "only direct calls to whitelisted functions are allowed")
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise DomainError(
                ["formula"],
                f"function '{node.func.id}' is not allowed; allowed: {', '.join(sorted(_FUNCTIONS))}",
            )
        if len(node.args) != 1:
            raise DomainError(["formula"], f"function '{node.func.id}' takes exactly one argument")
        return func(_convert(node.args[0], symbols, counter))

    if isinstance(node, ast.Name):
        if node.id in symbols:
            return symbols[node.id]
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise DomainError(["formula"], f"unknown symbol '{node.id}'")

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DomainError(["formula"], "only numeric literals are allowed")
        if not math.isfinite(value) or abs(value) > _MAX_LITERAL:
            raise DomainError(["formula"], f"numeric literal out of range (|x| <= {_MAX_LITERAL:g})")
        return sp.Float(value) if isinstance(value, float) else sp.Integer(value)

    raise DomainError(["formula"], f"unsupported syntax element '{type(node).__name__}'")


# ---------------------------------------------------------------------------
# Unit-aware evaluation of SymPy expressions
# ---------------------------------------------------------------------------

# SymPy function class -> numeric implementation for dimensionless arguments.
_FUNC_EVAL = {
    sp.sin: math.sin,
    sp.cos: math.cos,
    sp.tan: math.tan,
    sp.asin: math.asin,
    sp.acos: math.acos,
    sp.atan: math.atan,
    sp.exp: math.exp,
    sp.log: math.log,
}


def unit_str(unit) -> str:
    return "1" if unit == ureg.dimensionless else f"{unit:~C}"


def _describe(value) -> str:
    if isinstance(value, pint.Quantity):
        return f"unit '{unit_str(value.units)}'"
    return "dimensionless number"


def _as_dimensionless(value: pint.Quantity, what: str) -> float:
    if not value.dimensionless:
        raise DomainError(["formula"], f"{what} must be dimensionless, got unit '{unit_str(value.units)}'")
    return value.to(ureg.dimensionless).magnitude


def _eval(expr: sp.Expr, env: dict[str, pint.Quantity]):
    """Evaluate a SymPy expression with Pint quantities; returns float or Quantity."""
    if isinstance(expr, sp.Symbol):
        return env[expr.name]
    if expr.is_Number:
        return float(expr)

    if isinstance(expr, sp.Add):
        args = iter(expr.args)
        acc = _eval(next(args), env)
        for arg in args:
            value = _eval(arg, env)
            try:
                acc = acc + value
            except pint.DimensionalityError:
                raise DomainError(
                    ["formula"],
                    f"dimension mismatch: cannot add/subtract {_describe(acc)} and {_describe(value)}",
                ) from None
        return acc

    if isinstance(expr, sp.Mul):
        acc = 1.0
        for arg in expr.args:
            acc = acc * _eval(arg, env)
        return acc

    if isinstance(expr, sp.Pow):
        base = _eval(expr.base, env)
        exponent = _eval(expr.exp, env)
        if isinstance(exponent, pint.Quantity):
            exponent = _as_dimensionless(exponent, "exponent")
        try:
            return base ** exponent
        except (OverflowError, ZeroDivisionError) as exc:
            raise DomainError(["formula"], f"evaluation failed: {exc}") from None

    if isinstance(expr, sp.Abs):
        return abs(_eval(expr.args[0], env))

    if isinstance(expr, sp.sign):
        # May appear as the derivative of abs(); dimensionless by definition.
        value = _eval(expr.args[0], env)
        magnitude = value.magnitude if isinstance(value, pint.Quantity) else value
        return float((magnitude > 0) - (magnitude < 0))

    if isinstance(expr, sp.Function):
        func = _FUNC_EVAL.get(type(expr))
        name = type(expr).__name__
        if func is None:
            raise DomainError(["formula"], f"function '{name}' is not supported")
        argument = _eval(expr.args[0], env)
        if isinstance(argument, pint.Quantity):
            argument = _as_dimensionless(argument, f"argument of '{name}'")
        try:
            return func(argument)
        except (ValueError, OverflowError) as exc:
            raise DomainError(["formula"], f"function '{name}' failed: {exc}") from None

    raise DomainError(["formula"], f"cannot evaluate expression '{expr}'")


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

_TOL = 1e-9


def _correlation_matrix(names: list[str], correlation, n: int) -> np.ndarray:
    if correlation is None:
        return np.eye(n)

    order, matrix = correlation.order, correlation.matrix
    if len(order) != n or sorted(order) != sorted(names):
        raise DomainError(
            ["correlation", "order"],
            "must list every quantity name exactly once",
        )
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise DomainError(["correlation", "matrix"], f"matrix must be {n}x{n}")

    M = np.asarray(matrix, dtype=float)
    if not np.isfinite(M).all():
        raise DomainError(["correlation", "matrix"], "entries must be finite numbers")

    for i in range(n):
        if abs(M[i, i] - 1.0) > _TOL:
            raise DomainError(["correlation", "matrix", i, i], "diagonal entries must be 1")
    for i in range(n):
        for j in range(i + 1, n):
            if abs(M[i, j] - M[j, i]) > _TOL:
                raise DomainError(
                    ["correlation", "matrix", i, j],
                    f"matrix is not symmetric: entry [{i}][{j}]={M[i, j]:g} "
                    f"differs from [{j}][{i}]={M[j, i]:g}",
                )
            if abs(M[i, j]) > 1.0:
                raise DomainError(
                    ["correlation", "matrix", i, j],
                    "correlation coefficient must be within [-1, 1]",
                )

    min_eig = float(np.linalg.eigvalsh(M).min())
    if min_eig < -_TOL:
        raise DomainError(
            ["correlation", "matrix"],
            f"matrix is not positive semi-definite (smallest eigenvalue {min_eig:.6g})",
        )

    # Reorder from the submitted order to the quantity declaration order.
    perm = [order.index(name) for name in names]
    return M[np.ix_(perm, perm)]


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

@dataclass
class QuantitySpec:
    name: str
    value: float
    unit: str
    uncertainty: float


def _check_real_finite(value, loc: list, what: str) -> None:
    if isinstance(value, complex):
        raise DomainError(loc, f"{what} is complex; only real results are supported")
    if not math.isfinite(value):
        raise DomainError(loc, f"{what} is not finite")


def _magnitude_in(value, unit, loc: list) -> float:
    """Magnitude of a quantity expressed in ``unit`` (a structural zero maps to 0.0)."""
    try:
        magnitude = value.to(unit).magnitude
    except pint.DimensionalityError:
        if value.magnitude == 0:
            return 0.0
        raise DomainError(
            loc,
            f"cannot express contribution with unit '{unit_str(value.units)}' "
            f"in result unit '{unit_str(unit)}'",
        ) from None
    _check_real_finite(magnitude, loc, "contribution")
    return float(magnitude)


def propagate(quantities: list[QuantitySpec], formula: str, correlation) -> dict:
    names = [q.name for q in quantities]
    symbols = {name: sp.Symbol(name, real=True) for name in names}
    expr = parse_formula(formula, symbols)

    env: dict[str, pint.Quantity] = {}
    uenv: dict[str, pint.Quantity] = {}
    for i, q in enumerate(quantities):
        try:
            unit = ureg.parse_units(q.unit.strip() or "dimensionless")
        except pint.errors.UndefinedUnitError:
            raise DomainError(["quantities", i, "unit"], f"unknown unit '{q.unit}'") from None
        env[q.name] = Q_(q.value, unit)
        uenv[q.name] = Q_(q.uncertainty, unit)

    try:
        result = _eval(expr, env)
        derivatives = [_eval(sp.diff(expr, symbols[q.name]), env) for q in quantities]
    except DomainError:
        raise
    except pint.errors.PintError as exc:
        raise DomainError(["formula"], f"unit error: {exc}") from None

    if not isinstance(result, pint.Quantity):
        result = Q_(result, ureg.dimensionless)
    if result.dimensionless:
        # Reduce scaled dimensionless units (e.g. g/kg, percent) to a pure number.
        result = result.to(ureg.dimensionless)
    _check_real_finite(result.magnitude, ["formula"], "result value")
    y_unit = result.units

    # Sensitivity coefficients c_i and uncertainty influences t_i = c_i * u(x_i)
    # expressed in the result unit.
    sensitivities: list[float] = []
    influences: list[float] = []
    for i, q in enumerate(quantities):
        c = derivatives[i]
        c_qty = c if isinstance(c, pint.Quantity) else Q_(c, ureg.dimensionless)
        s_unit = y_unit / env[q.name].units
        try:
            s_mag = c_qty.to(s_unit).magnitude
        except pint.DimensionalityError:
            if c_qty.magnitude == 0:
                s_mag = 0.0  # structurally zero derivative carries no units
            else:
                raise DomainError(
                    ["formula"],
                    f"sensitivity for '{q.name}' has inconsistent units",
                ) from None
        _check_real_finite(s_mag, ["formula"], f"sensitivity for '{q.name}'")
        sensitivities.append(float(s_mag))
        influences.append(_magnitude_in(c_qty * uenv[q.name], y_unit, ["formula"]))

    n = len(quantities)
    R = _correlation_matrix(names, correlation, n)

    t = np.asarray(influences, dtype=float)
    Rt = R @ t
    variance = float(t @ Rt)
    variance = variance if variance > 0.0 else 0.0  # guard tiny negative round-off
    combined = math.sqrt(variance)
    _check_real_finite(combined, ["formula"], "combined standard uncertainty")

    contributions = []
    for i, q in enumerate(quantities):
        var_contrib = float(t[i] * Rt[i])  # own variance + correlation terms with all inputs
        contributions.append(
            {
                "name": q.name,
                "value": q.value,
                "unit": unit_str(env[q.name].units),
                "uncertainty": q.uncertainty,
                "sensitivity": sensitivities[i],
                "sensitivity_unit": unit_str(y_unit / env[q.name].units),
                "standard_contribution": abs(float(t[i])),
                "variance_contribution": var_contrib,
                "variance_share": (var_contrib / variance) if variance > 0.0 else None,
            }
        )

    return {
        "formula": formula,
        "value": float(result.magnitude),
        "unit": unit_str(y_unit),
        "combined_standard_uncertainty": combined,
        "variance": variance,
        "contributions": contributions,
    }
