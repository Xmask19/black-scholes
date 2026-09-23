"""
Black-Scholes option pricing.

Stage 1: call price for a European option on a non-dividend-paying stock.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt, exp
from scipy.stats import norm


def _d1_d2(
    S: float, K: float, t: float, r: float, sigma: float
) -> tuple[float, float]:
    """Return (d1, d2) for the Black-Scholes formula."""
    d1 = (log(S / K) + (r + sigma**2 / 2) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    return d1, d2


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
    d1, d2 = _d1_d2(S, K, t, r, sigma)
    return S * norm.cdf(d1) - K * exp(-r * t) * norm.cdf(d2)


def put_price(S: float, K: float, t: float, r: float, sigma: float) -> float:
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


def delta_call(S: float, K: float, t: float, r: float, sigma: float) -> float:
    """Delta of a European call: dC/dS = N(d1)."""
    d1, _ = _d1_d2(S, K, t, r, sigma)
    return norm.cdf(d1)


def delta_put(S: float, K: float, t: float, r: float, sigma: float) -> float:
    """Delta of a European put: dP/dS = N(d1) - 1."""
    d1, _ = _d1_d2(S, K, t, r, sigma)
    return norm.cdf(d1) - 1


def gamma(S: float, K: float, t: float, r: float, sigma: float) -> float:
    """Gamma: d^2V/dS^2 = N'(d1)/(S*sigma*sqrt(t))"""
    d1, _ = _d1_d2(S, K, t, r, sigma)
    return norm.pdf(d1) / (S * sigma * sqrt(t))


def vega(S: float, K: float, t: float, r: float, sigma: float) -> float:
    """Vega: dV/dsigma = SN'(d1)*t"""
    d1, _ = _d1_d2(S, K, t, r, sigma)
    return S * norm.pdf(d1) * sqrt(t)


def theta_call(S: float, K: float, t: float, r: float, sigma: float) -> float:
    """Theta of a European call.

    dC/dt = (-SN'(d1)sigma/(2sqrt(t))) - N(d2)rKe^(-rt)."""
    d1, d2 = _d1_d2(S, K, t, r, sigma)
    return ((-S * norm.pdf(d1) * sigma / (2 * sqrt(t))) -
            r * K * exp(-r * t) * norm.cdf(d2))


def theta_put(S: float, K: float, t: float, r: float, sigma: float) -> float:
    """Theta of a European put.

    dP/dt = (-SN'(d1)sigma/(2sqrt(t))) + N(-d2)rKe^(-rt)."""
    d1, d2 = _d1_d2(S, K, t, r, sigma)
    return ((-S * norm.pdf(d1) * sigma / (2 * sqrt(t))) +
            r * K * exp(-r * t) * norm.cdf(-d2))


def rho_call(S: float, K: float, t: float, r: float, sigma: float) -> float:
    """Rho of a European call: dC/dr = N(d2)Kte^-rt"""
    _, d2 = _d1_d2(S, K, t, r, sigma)
    return K * t * exp(-r * t) * norm.cdf(d2)


def rho_put(S: float, K: float, t: float, r: float, sigma: float) -> float:
    """Rho of a European put: dP/dr = -N(-d2)Kte^-rt"""
    _, d2 = _d1_d2(S, K, t, r, sigma)
    return -K * t * exp(-r * t) * norm.cdf(-d2)


def plot_price_vs_spot(
        K: float = 100, t: float = 1, r: float = 0.05, sigma: float = 0.2
) -> None:

    """
    Plot call and put prices against spot price S.

    Saves the figure to output/price_vs_spot.png.
    """
    spots = np.linspace(50, 150, 200)

    calls = [call_price(S, K, t, r, sigma) for S in spots]
    puts = [put_price(S, K, t, r, sigma) for S in spots]

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


def plot_greeks(K: float = 100, t: float = 1,
                r: float = 0.05, sigma: float = 0.2) -> None:
    """
    Plot delta, gamma, vega, theta, rho against spot price S.

    Saves the figure to output/greeks.png."""

    spots = np.linspace(50, 150, 200)

    deltas_call = [delta_call(S, K, t, r, sigma) for S in spots]
    deltas_put = [delta_put(S, K, t, r, sigma) for S in spots]
    gammas = [gamma(S, K, t, r, sigma) for S in spots]
    vegas = [vega(S, K, t, r, sigma) for S in spots]
    thetas_call = [theta_call(S, K, t, r, sigma) for S in spots]
    thetas_put = [theta_put(S, K, t, r, sigma) for S in spots]
    rhos_call = [rho_call(S, K, t, r, sigma) for S in spots]
    rhos_put = [rho_put(S, K, t, r, sigma) for S in spots]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for ax in axes.flat:
        ax.set_xlabel("Spot price S")
        ax.axvline(K, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(True)

    axes[0, 0].plot(spots, deltas_call, label="Call")
    axes[0, 0].plot(spots, deltas_put, label="Put")
    axes[0, 0].set_title("Delta")
    axes[0, 0].legend()

    axes[0, 1].plot(spots, gammas)
    axes[0, 1].set_title("Gamma")

    axes[0, 2].plot(spots, vegas)
    axes[0, 2].set_title("Vega")

    axes[1, 0].plot(spots, thetas_call, label="Call")
    axes[1, 0].plot(spots, thetas_put, label="Put")
    axes[1, 0].set_title("Theta")
    axes[1, 0].legend()

    axes[1, 1].plot(spots, rhos_call, label="Call")
    axes[1, 1].plot(spots, rhos_put, label="Put")
    axes[1, 1].set_title("Rho")
    axes[1, 1].legend()

    axes[1, 2].axis("off")

    fig.suptitle("Black-Scholes Greeks vs spot")
    fig.tight_layout()

    os.makedirs("output", exist_ok=True)
    plt.savefig("output/greeks.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # price = put_price(100, 100, 1, 0.05, 0.2)
    # print(f"Call price: {price:.4f}")

    K, t, r, sigma = 100, 1, 0.05, 0.2
    # plot_price_vs_spot(K=K, t=t, r=r, sigma=sigma)
    plot_greeks(K=K, t=t, r=r, sigma=sigma)
