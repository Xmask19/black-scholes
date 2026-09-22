"""
Black-Scholes option pricing.

Stage 1: call price for a European option on a non-dividend-paying stock.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
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


def put_price(S: float, K: float, t: float, r: float, sigma: float):
    """
    Return the Black-Scholes price of a European put option
    using put-call parity.

    P = C - S + K * exp(-r * T)

    Args:
        S: Current stock price.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free interest rate (annual, decimal).
        sigma: Volatility (annual, decimal).

    Returns:
        The put price.
    """
    return call_price(S, K, t, r, sigma) - S + K * exp(-r * t)


def plot_price_vs_spot(K: float, T: float, r: float, sigma: float) -> None:

    """
    Plot call and put prices against spot price S.

    Saves the figure to output/price_vs_spot.png.
    """
    spots = np.linspace(50, 150, 200)

    calls = [call_price(S, K, T, r, sigma) for S in spots]
    puts = [put_price(S, K, T, r, sigma) for S in spots]

    plt.figure(figsize=(8, 5))
    plt.plot(spots, calls, label="Call")
    plt.plot(spots, puts, label="Put")
    plt.axvline(K, color="gray", linestyle="--", label="Strike K")

    plt.xlabel("Spot price S")
    plt.ylabel("Option price")
    plt.title("Black-Scholes European option prices vs spot")
    plt.legend()
    plt.grid(True)

    os.makedirs("output", exist_ok=True)
    plt.savefig("output/price_vs_spot.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # price = put_price(100, 100, 1, 0.05, 0.2)
    # print(f"Call price: {price:.4f}")

    K, T, r, sigma = 100, 1, 0.05, 0.2
    plot_price_vs_spot(K=K, T=T, r=r, sigma=sigma)
