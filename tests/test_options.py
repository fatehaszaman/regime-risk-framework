"""Black-Scholes pricing, Greeks and Greeks-based P&L attribution.

Reference values are the standard textbook case S=100, K=100, r=5%, sigma=20%,
T=1 (Hull, Options, Futures, and Other Derivatives): call 10.4506, put 5.5735,
call delta 0.6368, gamma 0.01876, vega 37.52, call theta -6.414 per year.
"""
from math import exp

import pytest

from regime_risk import attribute, bs_greeks, bs_price

S, K, R, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0


# ------------------------------------------------------------------- pricing

def test_call_matches_the_textbook_value():
    assert bs_price(S, K, T, R, VOL, "call") == pytest.approx(10.4506, abs=1e-4)


def test_put_matches_the_textbook_value():
    assert bs_price(S, K, T, R, VOL, "put") == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("spot, strike, expiry", [(100.0, 100.0, 1.0), (80.0, 110.0, 0.25)])
def test_put_call_parity_holds(spot, strike, expiry):
    """C - P = S - K e^(-rT), independent of volatility."""
    c = bs_price(spot, strike, expiry, R, VOL, "call")
    p = bs_price(spot, strike, expiry, R, VOL, "put")
    assert c - p == pytest.approx(spot - strike * exp(-R * expiry), abs=1e-10)


# -------------------------------------------------------------------- Greeks

def test_call_greeks_match_the_textbook_values():
    g = bs_greeks(S, K, T, R, VOL, "call")
    assert g.delta == pytest.approx(0.6368, abs=1e-4)
    assert g.gamma == pytest.approx(0.01876, abs=1e-5)
    assert g.vega == pytest.approx(37.524, abs=1e-3)
    assert g.theta == pytest.approx(-6.414, abs=1e-3)


def test_put_delta_is_call_delta_minus_one_and_put_theta_matches():
    c, p = bs_greeks(S, K, T, R, VOL, "call"), bs_greeks(S, K, T, R, VOL, "put")
    assert p.delta == pytest.approx(c.delta - 1.0, abs=1e-12)
    assert p.theta == pytest.approx(-1.658, abs=1e-3)


@pytest.mark.parametrize("spot", [70.0, 130.0])
def test_delta_bounds(spot):
    """Call delta lies in (0, 1) and put delta in (-1, 0), deep in or out of the money."""
    assert 0.0 < bs_greeks(spot, K, T, R, VOL, "call").delta < 1.0
    assert -1.0 < bs_greeks(spot, K, T, R, VOL, "put").delta < 0.0


def test_gamma_and_vega_are_positive_and_equal_for_calls_and_puts():
    c, p = bs_greeks(S, K, T, R, VOL, "call"), bs_greeks(S, K, T, R, VOL, "put")
    assert c.gamma > 0 and c.vega > 0
    assert p.gamma == pytest.approx(c.gamma, abs=1e-12)
    assert p.vega == pytest.approx(c.vega, abs=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_analytic_greeks_match_finite_differences(kind):
    """Bump-and-reprice check, so a sign or factor error in a formula fails.
    Uses T = 0.5 and an out-of-the-money strike, where sqrt(T) and ln(S/K)
    are not 1 and 0 and so cannot hide a missing term."""
    s0, k, t0 = 100.0, 110.0, 0.5
    g = bs_greeks(s0, k, t0, R, VOL, kind)
    h, hv, ht = 0.01, 1e-5, 1e-6

    def px(s=s0, vol=VOL, t=t0):
        return bs_price(s, k, t, R, vol, kind)

    assert g.delta == pytest.approx((px(s=s0 + h) - px(s=s0 - h)) / (2 * h), abs=1e-6)
    assert g.gamma == pytest.approx((px(s=s0 + h) - 2 * px() + px(s=s0 - h)) / h**2, abs=1e-4)
    assert g.vega == pytest.approx((px(vol=VOL + hv) - px(vol=VOL - hv)) / (2 * hv), abs=1e-4)
    # Theta is the value change per year of elapsed time, so it is -dV/dT.
    assert g.theta == pytest.approx(-(px(t=t0 + ht) - px(t=t0 - ht)) / (2 * ht), abs=1e-4)


# --------------------------------------------------------------- attribution

def test_no_volatility_change_gives_zero_vega_pnl():
    a = attribute(S, 105.0, VOL, VOL, 10 / 365, K, T, R, "call")
    assert a.vega == 0.0


def test_no_price_change_gives_zero_delta_and_gamma_pnl():
    a = attribute(S, S, VOL, 0.25, 10 / 365, K, T, R, "call")
    assert a.delta == 0.0
    assert a.gamma == 0.0


@pytest.mark.parametrize("kind, quantity", [("call", 100.0), ("put", -250.0)])
def test_parts_sum_to_the_total(kind, quantity):
    """Same reconciliation as the scenario engine: the parts, including the
    residual, add up to the full revaluation."""
    a = attribute(S, 93.0, VOL, 0.27, 30 / 365, K, T, R, kind, quantity)
    full = quantity * (bs_price(93.0, K, T - 30 / 365, R, 0.27, kind)
                       - bs_price(S, K, T, R, VOL, kind))
    assert a.total == pytest.approx(full, abs=1e-9)
    assert sum(a.parts().values()) == pytest.approx(a.total, abs=1e-9)


def test_residual_shrinks_as_the_move_shrinks():
    """The residual holds second-order cross terms and higher, so halving
    dS, d_sigma and dt should cut it by roughly four."""
    def residual(h):
        return abs(attribute(S, S + 4 * h, VOL, VOL + 0.02 * h, (10 / 365) * h,
                             K, T, R, "call").residual)

    r1, r2, r3 = residual(1.0), residual(0.5), residual(0.25)
    assert r2 < 0.3 * r1
    assert r3 < 0.3 * r2

    # With only the spot moving, delta and gamma capture the first two Taylor
    # terms, so the residual is third order and halving dS cuts it by about 8.
    def spot_only(dS):
        return abs(attribute(S, S + dS, VOL, VOL, 0.0, K, T, R, "call").residual)

    assert spot_only(2.0) < 0.2 * spot_only(4.0)


def test_short_position_mirrors_the_long_position():
    long_ = attribute(S, 104.0, VOL, 0.22, 5 / 365, K, T, R, "call", 10.0)
    short = attribute(S, 104.0, VOL, 0.22, 5 / 365, K, T, R, "call", -10.0)
    for key, value in long_.parts().items():
        assert short.parts()[key] == pytest.approx(-value, abs=1e-12)


@pytest.mark.parametrize("args", [
    (0.0, K, T, R, VOL, "call"),      # non-positive spot
    (S, K, T, R, 0.0, "call"),        # zero volatility
    (S, K, T, R, VOL, "straddle"),    # unknown kind
])
def test_invalid_inputs_are_rejected(args):
    with pytest.raises(ValueError):
        bs_price(*args)


def test_horizon_reaching_expiry_is_rejected():
    with pytest.raises(ValueError, match="shorter than time to expiry"):
        attribute(S, S, VOL, VOL, 1.0, K, 1.0, R, "call")
