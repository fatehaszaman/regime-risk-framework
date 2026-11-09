"""
options.py
----------
Black-Scholes pricing, Greeks and Greeks-based P&L attribution for European
options on a non-dividend-paying underlying.

Notation: S spot, K strike, T years to expiry, r continuously compounded
risk-free rate, sigma annualized volatility, N the standard normal CDF and
phi its density.

    d1 = [ln(S/K) + (r + sigma^2 / 2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)
    call = S N(d1) - K e^(-rT) N(d2)
    put  = K e^(-rT) N(-d2) - S N(-d1)

Greeks, per unit of the underlying:

    delta  call N(d1), put N(d1) - 1
    gamma  phi(d1) / (S sigma sqrt(T))                      calls and puts
    vega   S phi(d1) sqrt(T), per 1.00 of volatility        calls and puts
    theta  call -S phi(d1) sigma / (2 sqrt(T)) - r K e^(-rT) N(d2)
           put  -S phi(d1) sigma / (2 sqrt(T)) + r K e^(-rT) N(-d2)

Theta is per year of elapsed time. Elapsed time uses calendar days, so a
one-day horizon is dt = 1/365; a trading-day convention (days/252) would give
a larger theta P&L for the same horizon.

Attribution over a move from (S0, sigma0, T) to (S1, sigma1, T - dt):

    delta P&L    = delta * dS
    gamma P&L    = 1/2 * gamma * dS^2
    vega P&L     = vega * d_sigma
    theta P&L    = theta * dt
    residual     = actual change in value - the four terms above

The residual holds the higher-order and cross terms (vanna, volga, the
change in delta over the horizon, and so on). It is reported rather than
hidden, so the parts always sum to the full revaluation.

Scope: European exercise, no dividends, constant r over the horizon. This is
a sensitivity tool for scenario analysis, not a calibrated pricing library.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, isfinite, log, pi, sqrt

_KINDS = ("call", "put")


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _validate(S: float, K: float, T: float, r: float, sigma: float, kind: str) -> None:
    if kind not in _KINDS:
        raise ValueError("Option kind must be 'call' or 'put'")
    if not all(isfinite(x) for x in (S, K, T, r, sigma)):
        raise ValueError("Option inputs must be finite")
    if S <= 0 or K <= 0:
        raise ValueError("Spot and strike must be positive")
    if T <= 0:
        raise ValueError("Time to expiry must be positive")
    if sigma <= 0:
        raise ValueError("Volatility must be positive")


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    vol_sqrt_t = sigma * sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(S: float, K: float, T: float, r: float, sigma: float, kind: str = "call") -> float:
    """Black-Scholes value of one European option on one unit of the underlying."""
    _validate(S, K, T, r, sigma, kind)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    discounted_k = K * exp(-r * T)
    if kind == "call":
        return S * norm_cdf(d1) - discounted_k * norm_cdf(d2)
    return discounted_k * norm_cdf(-d2) - S * norm_cdf(-d1)


@dataclass(frozen=True)
class Greeks:
    """First-order Greeks plus gamma, per unit of the underlying."""
    delta: float
    gamma: float
    vega: float     # per 1.00 change in volatility (0.01 = one vol point)
    theta: float    # per year of elapsed time


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, kind: str = "call") -> Greeks:
    """Analytic Black-Scholes delta, gamma, vega and theta."""
    _validate(S, K, T, r, sigma, kind)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    sqrt_t = sqrt(T)
    pdf_d1 = norm_pdf(d1)
    discounted_k = K * exp(-r * T)

    gamma = pdf_d1 / (S * sigma * sqrt_t)
    vega = S * pdf_d1 * sqrt_t
    decay = -S * pdf_d1 * sigma / (2.0 * sqrt_t)
    if kind == "call":
        delta = norm_cdf(d1)
        theta = decay - r * discounted_k * norm_cdf(d2)
    else:
        delta = norm_cdf(d1) - 1.0
        theta = decay + r * discounted_k * norm_cdf(-d2)
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta)


@dataclass(frozen=True)
class GreeksAttribution:
    """Change in option value split into Greek terms and a residual."""
    delta: float
    gamma: float
    vega: float
    theta: float
    residual: float
    total: float    # full revaluation: value after the move minus value before

    def parts(self) -> dict[str, float]:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "residual": self.residual,
        }


def attribute(
    S0: float,
    S1: float,
    vol0: float,
    vol1: float,
    dt: float,
    K: float,
    T: float,
    r: float,
    kind: str = "call",
    quantity: float = 1.0,
) -> GreeksAttribution:
    """
    Attribute the change in value of `quantity` options to delta, gamma, vega
    and theta, measured with Greeks at the starting point (S0, vol0, T).

    `quantity` is signed: positive for a long position, negative for short.
    `dt` is the elapsed time in years and must leave time to expiry
    (0 <= dt < T); expiry within the horizon is outside the model.
    """
    if not isfinite(quantity):
        raise ValueError("Quantity must be finite")
    if not isfinite(dt) or dt < 0:
        raise ValueError("Elapsed time must be finite and non-negative")
    if dt >= T:
        raise ValueError("Elapsed time must be shorter than time to expiry")

    v0 = bs_price(S0, K, T, r, vol0, kind)
    v1 = bs_price(S1, K, T - dt, r, vol1, kind)
    g = bs_greeks(S0, K, T, r, vol0, kind)

    dS = S1 - S0
    delta_pnl = quantity * g.delta * dS
    gamma_pnl = quantity * 0.5 * g.gamma * dS * dS
    vega_pnl = quantity * g.vega * (vol1 - vol0)
    theta_pnl = quantity * g.theta * dt
    total = quantity * (v1 - v0)
    residual = total - (delta_pnl + gamma_pnl + vega_pnl + theta_pnl)
    return GreeksAttribution(
        delta=delta_pnl,
        gamma=gamma_pnl,
        vega=vega_pnl,
        theta=theta_pnl,
        residual=residual,
        total=total,
    )
