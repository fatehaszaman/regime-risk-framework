"""Fixtures built so the correct answer can be worked out by hand."""
import pandas as pd
import pytest

from regime_risk import LCSettlement, PendingLC, Position

D0 = pd.Timestamp("2024-01-01")


def settlement(day, official, realized, amount=1_000_000.0, commodity="copper"):
    return LCSettlement(
        settlement_date=D0 + pd.Timedelta(days=day),
        official_rate=official,
        realized_rate=realized,
        lc_amount_usd=amount,
        commodity=commodity,
        counterparty_bank="BankA",
        settlement_days=30,
    )


@pytest.fixture
def flat_settlements():
    """Official flat at 100, realized flat at 103. True basis is exactly 3.0."""
    return [settlement(d, 100.0, 103.0) for d in range(12) for _ in range(3)]


@pytest.fixture
def zero_basis_trending():
    """Realized EQUALS official every day, official trending up 1.0/day.

    The true basis is identically zero. Any non-zero basis the builder reports
    is an artefact of its own smoothing.
    """
    return [settlement(d, 100.0 + d, 100.0 + d) for d in range(12) for _ in range(3)]


def lc(lc_id, amount, urgency=5.0, strategic=5.0, days=60, unit_cost=100.0):
    return PendingLC(
        lc_id=lc_id,
        commodity="copper",
        amount_usd=amount,
        urgency_score=urgency,
        strategic_score=strategic,
        days_until_expiry=days,
        unit_cost_usd=unit_cost,
        quantity=amount / unit_cost if unit_cost else amount,
        unit="tonne",
    )


@pytest.fixture
def make_lc():
    return lc


def position(pid="P1", notional=1_000_000.0, commodity="copper", settlement_fx=110.0):
    return Position(
        position_id=pid,
        commodity=commodity,
        notional_usd=notional,
        quantity=1000.0,
        unit="tonne",
        entry_price_usd=1000.0,
        lc_fee_pct=0.01,
        tariff_rate=0.05,
        settlement_fx=settlement_fx,
    )


@pytest.fixture
def make_position():
    return position


@pytest.fixture
def one_position():
    return [position()]
