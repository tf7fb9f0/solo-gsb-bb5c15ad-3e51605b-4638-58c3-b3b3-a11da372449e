"""End-to-end tests for the uncertainty propagation API."""
import math

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

DENSITY = {
    "quantities": [
        {"name": "m", "value": 100.0, "unit": "g", "uncertainty": 0.05},
        {"name": "V", "value": 100.0, "unit": "mL", "uncertainty": 0.1},
    ],
    "formula": "m / V",
}


def post(payload):
    return client.post("/v1/propagate", json=payload)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_density_uncorrelated():
    resp = post(DENSITY)
    assert resp.status_code == 200
    body = resp.json()
    assert body["value"] == pytest.approx(1.0)
    assert body["unit"] == "g/ml"
    # t_m = 0.01*0.05 = 5e-4 ; t_V = 0.01*0.1 = 1e-3
    expected = math.sqrt(5e-4**2 + 1e-3**2)
    assert body["combined_standard_uncertainty"] == pytest.approx(expected)
    assert body["variance"] == pytest.approx(expected**2)
    contribs = {c["name"]: c for c in body["contributions"]}
    assert contribs["m"]["sensitivity"] == pytest.approx(0.01)
    assert contribs["m"]["sensitivity_unit"] == "1/ml"
    assert contribs["m"]["standard_contribution"] == pytest.approx(5e-4)
    assert contribs["m"]["variance_contribution"] == pytest.approx(2.5e-7)
    assert contribs["V"]["variance_contribution"] == pytest.approx(1e-6)
    total = sum(c["variance_contribution"] for c in body["contributions"])
    assert total == pytest.approx(body["variance"])


def test_correlation_terms_in_contributions():
    payload = dict(DENSITY, correlation={
        "order": ["m", "V"],
        "matrix": [[1.0, 0.5], [0.5, 1.0]],
    })
    body = post(payload).json()
    # t_m = 5e-4, t_V = -1e-3 (sensitivity of m/V wrt V is negative)
    var = 5e-4**2 + 1e-3**2 + 2 * 0.5 * 5e-4 * -1e-3
    assert body["variance"] == pytest.approx(var)
    contribs = {c["name"]: c for c in body["contributions"]}
    # row sums of the contribution matrix include the correlation term
    assert contribs["m"]["variance_contribution"] == pytest.approx(5e-4 * (5e-4 + 0.5 * -1e-3))
    assert contribs["V"]["variance_contribution"] == pytest.approx(-1e-3 * (-1e-3 + 0.5 * 5e-4))
    total = sum(c["variance_contribution"] for c in body["contributions"])
    assert total == pytest.approx(body["variance"])


def test_correlation_order_is_respected():
    payload = dict(DENSITY, correlation={
        "order": ["V", "m"],  # reversed on purpose
        "matrix": [[1.0, 0.5], [0.5, 1.0]],
    })
    body = post(payload).json()
    var = 5e-4**2 + 1e-3**2 + 2 * 0.5 * 5e-4 * -1e-3
    assert body["variance"] == pytest.approx(var)


def test_asymmetric_correlation_matrix_422():
    payload = dict(DENSITY, correlation={
        "order": ["m", "V"],
        "matrix": [[1.0, 0.3], [0.8, 1.0]],
    })
    resp = post(payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"][0]
    assert detail["loc"] == ["body", "correlation", "matrix", 0, 1]
    assert "symmetric" in detail["msg"]


def test_non_positive_semidefinite_matrix_422():
    payload = {
        "quantities": [
            {"name": "a", "value": 1.0, "unit": "g", "uncertainty": 0.1},
            {"name": "b", "value": 1.0, "unit": "g", "uncertainty": 0.1},
            {"name": "c", "value": 1.0, "unit": "g", "uncertainty": 0.1},
        ],
        "formula": "a + b + c",
        "correlation": {
            "order": ["a", "b", "c"],
            # symmetric, unit diagonal, |r| <= 1, but indefinite (det < 0)
            "matrix": [[1.0, 0.9, 0.9], [0.9, 1.0, 0.0], [0.9, 0.0, 1.0]],
        },
    }
    resp = post(payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"][0]
    assert detail["loc"] == ["body", "correlation", "matrix"]
    assert "positive semi-definite" in detail["msg"]


def test_correlation_order_mismatch_422():
    payload = dict(DENSITY, correlation={
        "order": ["m", "X"],
        "matrix": [[1.0, 0.0], [0.0, 1.0]],
    })
    resp = post(payload)
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "correlation", "order"]


def test_incompatible_units_422():
    payload = dict(DENSITY, formula="m + V")
    resp = post(payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"][0]
    assert detail["loc"] == ["body", "formula"]
    assert "dimension mismatch" in detail["msg"]


def test_unknown_unit_422():
    payload = {
        "quantities": [{"name": "x", "value": 1.0, "unit": "smoot", "uncertainty": 0.1}],
        "formula": "2 * x",
    }
    resp = post(payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"][0]
    assert detail["loc"] == ["body", "quantities", 0, "unit"]
    assert "unknown unit" in detail["msg"]


def test_unknown_symbol_422():
    resp = post(dict(DENSITY, formula="m / W"))
    assert resp.status_code == 422
    assert "unknown symbol 'W'" in resp.json()["detail"][0]["msg"]


@pytest.mark.parametrize("formula", [
    "m % V",
    "m // V",
    "m @ V",
    "m & V",
    "__import__('os')",
    "m.__class__",
    "[m, V]",
    "m if m else V",
    "f(m)",
    "m ** 1000",
])
def test_disallowed_constructs_422(formula):
    resp = post(dict(DENSITY, formula=formula))
    assert resp.status_code == 422, formula
    assert resp.json()["detail"][0]["loc"] == ["body", "formula"]


def test_dimensioned_function_argument_422():
    resp = post(dict(DENSITY, formula="sin(m)"))
    assert resp.status_code == 422
    assert "dimensionless" in resp.json()["detail"][0]["msg"]


def test_whitelisted_functions_and_constants():
    payload = {
        "quantities": [
            {"name": "x", "value": 2.0, "unit": "", "uncertainty": 0.01},
            {"name": "y", "value": 9.0, "unit": "m", "uncertainty": 0.09},
        ],
        "formula": "sqrt(y) * (exp(ln(x)) + abs(-x) + sin(pi / 2) * x)",
    }
    body = post(payload).json()
    # value: 3 * (2 + 2 + 1*2) = 18
    assert body["value"] == pytest.approx(18.0)
    assert body["unit"] == "m**0.5"
    # c_x = 3 * (1 + 1 + 1) = 9, c_y = 0.5/sqrt(9) * 6 = 1
    assert body["combined_standard_uncertainty"] == pytest.approx(
        math.sqrt((9 * 0.01) ** 2 + (1 * 0.09) ** 2)
    )


def test_dimensionless_result():
    payload = {
        "quantities": [
            {"name": "a", "value": 2.0, "unit": "g", "uncertainty": 0.1},
            {"name": "b", "value": 4.0, "unit": "kg", "uncertainty": 0.001},
        ],
        "formula": "a / b",
    }
    body = post(payload).json()
    assert body["value"] == pytest.approx(5e-4)
    assert body["unit"] == "1"


def test_validation_error_format():
    resp = post({"quantities": [], "formula": ""})
    assert resp.status_code == 422
    assert "detail" in resp.json()


def test_negative_uncertainty_422():
    payload = {
        "quantities": [{"name": "x", "value": 1.0, "unit": "g", "uncertainty": -0.1}],
        "formula": "x",
    }
    assert post(payload).status_code == 422


def test_duplicate_names_422():
    payload = {
        "quantities": [
            {"name": "x", "value": 1.0, "unit": "g", "uncertainty": 0.1},
            {"name": "x", "value": 2.0, "unit": "g", "uncertainty": 0.1},
        ],
        "formula": "x",
    }
    assert post(payload).status_code == 422
