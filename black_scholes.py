"""
Black-Scholes option pricing.

Stage 1: call price for a European option on a non-dividend-paying stock.
"""

from math import log, sqrt, exp
from scipy.stats import norm


def call_price(S: float, K: float, t: float, r: float, sigma: float) -> float:
    """
    Return the Black-Scholes price of a European call option.

    Args:
        S: Current stock price.
        K: Strike price.
        t: Time to maturity in years.
        r: Risk-free interest rate (annual, decimal).
        sigma: Volatility (annual, decimal).

    Returns:
        The call price.

    Example:
        >>> call_price(100, 100, 1, 0.05, 0.2)
        10.4506...
    """

    d1 = log(S/K) + (r + (sigma**2)/2) * t / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    return S * norm.cdf(d1) - K * exp(-r * t) * norm.cdf(d2)


if __name__ == "__main__":
    price = call_price(100, 100, 1, 0.05, 0.2)
    print(f"Call price: {price:.4f}")
