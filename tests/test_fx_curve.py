"""Effective-FX curve builder.

The basis is the headline output of this module and the largest-weighted input
to the regime classifier, so it gets tested against series whose true basis is
known by construction.
"""
import numpy as np
import pytest

from regime_risk import EffectiveFXCurveBuilder
from tests.conftest import settlement


def test_flat_series_recovers_the_true_basis(flat_settlements):
    """Official 100, realized 103, both flat. Basis must be 3.0 throughout."""
    curve = EffectiveFXCurveBuilder().build(flat_settlements)
    assert np.allclose(curve.basis, 3.0)


def test_effective_rate_is_volume_weighted_not_equal_weighted():
    """Two settlements on one day: 1m at 100 and 3m at 108.
    Volume-weighted mean is 106.0; the equal-weighted mean would be 104.0."""
    s = [settlement(0, 100.0, 100.0, amount=1_000_000.0),
         settlement(0, 100.0, 108.0, amount=3_000_000.0)]
    curve = EffectiveFXCurveBuilder(smoothing_window=1).build(s)
    assert curve.effective_rates[0] == pytest.approx(106.0)


def test_thin_days_fall_back_to_the_official_rate():
    """min_observations=2, so a day with one settlement uses the official rate
    and therefore reports zero basis."""
    s = [settlement(0, 100.0, 130.0)]
    curve = EffectiveFXCurveBuilder(min_observations=2, smoothing_window=1).build(s)
    assert curve.effective_rates[0] == pytest.approx(100.0)
    assert curve.basis[0] == pytest.approx(0.0)


def test_a_day_meeting_min_observations_uses_realized_rates():
    s = [settlement(0, 100.0, 130.0), settlement(0, 100.0, 130.0)]
    curve = EffectiveFXCurveBuilder(min_observations=2, smoothing_window=1).build(s)
    assert curve.effective_rates[0] == pytest.approx(130.0)


def test_regime_flag_fires_above_the_threshold():
    s = [settlement(d, 100.0, 105.0) for d in range(6) for _ in range(3)]
    curve = EffectiveFXCurveBuilder(stress_basis_threshold=2.0, smoothing_window=1).build(s)
    assert curve.regime_flags.all()


def test_regime_flag_stays_down_below_the_threshold():
    s = [settlement(d, 100.0, 101.0) for d in range(6) for _ in range(3)]
    curve = EffectiveFXCurveBuilder(stress_basis_threshold=2.0, smoothing_window=1).build(s)
    assert not curve.regime_flags.any()


def test_one_row_per_settlement_date(flat_settlements):
    curve = EffectiveFXCurveBuilder().build(flat_settlements)
    assert len(curve.dates) == 12
    assert len(curve.basis) == 12


def test_dates_come_back_sorted():
    s = [settlement(d, 100.0, 103.0) for d in (5, 1, 3, 0) for _ in range(3)]
    curve = EffectiveFXCurveBuilder().build(s)
    assert list(curve.dates) == sorted(curve.dates)


def test_summary_columns(flat_settlements):
    b = EffectiveFXCurveBuilder()
    df = b.summary(b.build(flat_settlements))
    assert list(df.columns) == ["date", "official_rate", "effective_rate", "basis", "stressed"]
    assert len(df) == 12


def test_basis_is_effective_minus_official(flat_settlements):
    curve = EffectiveFXCurveBuilder().build(flat_settlements)
    assert np.allclose(curve.basis, curve.effective_rates - curve.official_rates)


# ------------------------------------------------------------------- defects

def test_smoothing_preserves_zero_basis_when_official_rate_trends(zero_basis_trending):
    """Smoothing the basis must not turn a shared FX trend into a dislocation."""
    curve = EffectiveFXCurveBuilder(smoothing_window=5).build(zero_basis_trending)

    assert np.allclose(curve.basis, 0.0)
    assert np.allclose(curve.effective_rates, curve.official_rates)


def test_trending_official_rate_does_not_fire_false_stress_flags():
    """A falling official rate shared by realized settlements is not stress."""
    s = [settlement(d, 200.0 - 1.5 * d, 200.0 - 1.5 * d) for d in range(12) for _ in range(3)]
    curve = EffectiveFXCurveBuilder(smoothing_window=5, stress_basis_threshold=2.0).build(s)

    assert np.allclose(curve.basis, 0.0)
    assert not curve.regime_flags.any()


def test_smoothing_window_counts_rows_not_calendar_days():
    """DEFECT: smoothing_window is documented as "Rolling window (in days)" but
    `.rolling(n)` counts ROWS of the grouped frame, which are settlement dates.

    Settlement data is sparse and irregular, so a 5-row window can span weeks.
    Here five settlement dates are spread over 100 calendar days and are still
    averaged together as though adjacent. See KNOWN_ISSUES.md #2.
    """
    s = [settlement(d, 100.0, 100.0 + d) for d in (0, 25, 50, 75, 100) for _ in range(3)]
    curve = EffectiveFXCurveBuilder(smoothing_window=5).build(s)
    # Mean of realized 100,125,150,175,200 = 150 -> basis 50 against official 100
    assert curve.basis[-1] == pytest.approx(50.0)


def test_negative_basis_is_never_flagged_however_large():
    """DEFECT: regime_flags uses `basis > threshold`, a one-sided test.

    A large NEGATIVE basis — realized settling far below the official rate — is
    also a dislocation and is never flagged. Under import controls the basis is
    usually positive, which makes the one-sided test defensible, but the
    directional choice is not documented. See KNOWN_ISSUES.md #3.
    """
    s = [settlement(d, 100.0, 40.0) for d in range(6) for _ in range(3)]
    curve = EffectiveFXCurveBuilder(stress_basis_threshold=2.0, smoothing_window=1).build(s)
    assert curve.basis[-1] == pytest.approx(-60.0)
    assert not curve.regime_flags.any(), "a 60-unit dislocation goes unflagged"


def test_empty_settlement_list_raises_an_opaque_error():
    """DEFECT: there is no input validation. An empty list produces an empty
    DataFrame and fails on a missing column rather than saying so.
    See KNOWN_ISSUES.md #4.
    """
    with pytest.raises(KeyError):
        EffectiveFXCurveBuilder().build([])


def test_official_rate_is_equal_weighted_while_effective_is_volume_weighted():
    """DEFECT, low: within a date the official rate is aggregated with `.mean()`
    while the effective rate is volume-weighted.

    The official rate should be identical across settlements on a given day, so
    disagreement means bad input data — and it is silently averaged away rather
    than reported. See KNOWN_ISSUES.md #5.
    """
    s = [settlement(0, 100.0, 100.0, amount=1_000_000.0),
         settlement(0, 200.0, 100.0, amount=9_000_000.0)]
    curve = EffectiveFXCurveBuilder(smoothing_window=1).build(s)
    assert curve.official_rates[0] == pytest.approx(150.0), "equal-weighted, not 190"
