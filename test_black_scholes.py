"""
Known-value tests for the Black-Scholes pricing functions and Greeks.

Run with:
    pytest -v

Reference values are for S=100, K=100, T=1, r=0.05, sigma=0.2,
the standard Black-Scholes example.
"""

import pytest

from black_scholes import (
    call_price,
    put_price,
    delta_call,
    delta_put,
    gamma,
    vega,
    theta_call,
    theta_put,
    rho_call,
    rho_put,
)

from math import exp

# --- Test case ------------------------------------------------------
S0, K0, T0, R0, SIGMA0 = 100.0, 100.0, 1.0, 0.05, 0.2


# --- Prices ---------------------------------------------------------
def test_call_price():
    result = call_price(S0, K0, T0, R0, SIGMA0)
    assert result == pytest.approx(10.4506, abs=1e-3)


def test_put_price():
    result = put_price(S0, K0, T0, R0, SIGMA0)
    assert result == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity():
    c = call_price(S0, K0, T0, R0, SIGMA0)
    p = put_price(S0, K0, T0, R0, SIGMA0)
    assert c - p == pytest.approx(S0 - K0 * exp(-R0 * T0), abs=1e-9)


# --- Greeks ---------------------------------------------------------
def test_delta_call():
    result = delta_call(S0, K0, T0, R0, SIGMA0)
    assert result == pytest.approx(0.6368, abs=1e-4)


def test_delta_put():
    result = delta_put(S0, K0, T0, R0, SIGMA0)
    assert result == pytest.approx(-0.3632, abs=1e-4)


def test_gamma():
    result = gamma(S0, K0, T0, R0, SIGMA0)
    assert result == pytest.approx(0.018762, abs=1e-6)


def test_vega():
    result = vega(S0, K0, T0, R0, SIGMA0)
    assert result == pytest.approx(37.5240, abs=1e-3)


def test_theta_call():
    result = theta_call(S0, K0, T0, R0, SIGMA0)
    assert result == pytest.approx(-6.4140, abs=1e-3)


def test_theta_put():
    result = theta_put(S0, K0, T0, R0, SIGMA0)
    assert result == pytest.approx(-1.6579, abs=1e-3)


def test_rho_call():
    result = rho_call(S0, K0, T0, R0, SIGMA0)
    assert result == pytest.approx(53.2325, abs=1e-3)


def test_rho_put():
    result = rho_put(S0, K0, T0, R0, SIGMA0)
    assert result == pytest.approx(-41.8905, abs=1e-3)


# --- Identity that must hold exactly ------------------------------
def test_delta_parity():
    diff = (delta_call(S0, K0, T0, R0, SIGMA0) -
            delta_put(S0, K0, T0, R0, SIGMA0))
    assert diff == pytest.approx(1.0, abs=1e-12)
