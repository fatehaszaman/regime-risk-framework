"""Regime classifier.

The classifier turns four boolean signals into a label that downstream code
treats as "standard models unreliable". The weighting is arithmetic and fully
checkable by hand, so it is checked by hand.
"""
import numpy as np
import pandas as pd
import pytest

from regime_risk import RegimeClassifier, RegimeLabel

DATES = pd.date_range("2024-01-01", periods=10, freq="D")
CALM = np.zeros(10)
FLAT_VOL = np.full(10, 0.01)


def classify_one(**kw):
    """Single date, signals controlled directly."""
    fx = np.array([kw.get("fx", 0.0)])
    lc = np.array([kw.get("lc", 0.0)])
    vol = np.array([0.01])
    clf = RegimeClassifier(policy_event_dates=kw.get("policy_dates"))
    return clf.classify(pd.DatetimeIndex(["2024-01-01"]), fx, lc, vol)[0]


# ------------------------------------------------------------ weights, labels

def test_no_signals_scores_zero_and_labels_normal():
    sig = classify_one()
    assert sig.score == 0.0
    assert sig.label is RegimeLabel.NORMAL
    assert sig.notes == "No signals"


def test_fx_signal_alone_scores_its_documented_weight():
    sig = classify_one(fx=5.0)
    assert sig.score == pytest.approx(0.35)
    assert sig.label is RegimeLabel.ELEVATED


def test_lc_signal_alone_scores_its_documented_weight():
    sig = classify_one(lc=0.99)
    assert sig.score == pytest.approx(0.30)
    assert sig.label is RegimeLabel.ELEVATED


def test_policy_signal_alone_scores_its_documented_weight():
    sig = classify_one(policy_dates=["2024-01-01"])
    assert sig.score == pytest.approx(0.20)
    assert sig.label is RegimeLabel.ELEVATED


def test_weights_sum_to_one():
    assert sum(RegimeClassifier.SIGNAL_WEIGHTS.values()) == pytest.approx(1.0)


def test_fx_plus_lc_is_stressed_not_crisis():
    """0.35 + 0.30 = 0.65, inside the STRESSED band [0.45, 0.70)."""
    sig = classify_one(fx=5.0, lc=0.99)
    assert sig.score == pytest.approx(0.65)
    assert sig.label is RegimeLabel.STRESSED


def test_three_signals_reach_crisis():
    """0.35 + 0.30 + 0.20 = 0.85."""
    sig = classify_one(fx=5.0, lc=0.99, policy_dates=["2024-01-01"])
    assert sig.score == pytest.approx(0.85)
    assert sig.label is RegimeLabel.CRISIS


def test_all_four_signals_score_exactly_one():
    """The 21st observation can fire after the 20-observation cold start."""
    n = 21
    vol = np.full(n, 0.01)
    vol[-1] = 1.0
    clf = RegimeClassifier(policy_event_dates=["2024-01-21"])
    sig = clf.classify(pd.date_range("2024-01-01", periods=n), np.full(n, 5.0),
                       np.full(n, 0.99), vol)[-1]
    assert sig.volatility_signal is True
    assert sig.score == pytest.approx(1.0)
    assert sig.label is RegimeLabel.CRISIS


def test_label_boundaries_are_inclusive_at_the_lower_edge():
    clf = RegimeClassifier()
    assert clf._label(0.20) is RegimeLabel.ELEVATED
    assert clf._label(0.1999) is RegimeLabel.NORMAL
    assert clf._label(0.45) is RegimeLabel.STRESSED
    assert clf._label(0.70) is RegimeLabel.CRISIS


def test_thresholds_are_strict_so_a_value_on_the_line_does_not_fire():
    assert classify_one(fx=2.0).fx_basis_signal is False
    assert classify_one(fx=2.0001).fx_basis_signal is True


def test_notes_list_the_active_signals():
    sig = classify_one(fx=5.0, lc=0.99)
    assert "FX basis" in sig.notes and "LC utilization" in sig.notes


def test_one_signal_per_input_date():
    clf = RegimeClassifier()
    assert len(clf.classify(DATES, CALM, CALM, FLAT_VOL)) == 10


def test_to_dataframe_shape():
    clf = RegimeClassifier()
    df = clf.to_dataframe(clf.classify(DATES, CALM, CALM, FLAT_VOL))
    assert len(df) == 10
    assert "regime" in df.columns and "score" in df.columns


def test_constant_volatility_does_not_fire_the_vol_signal():
    """The epsilon guard on the standard deviation is correct here: a constant
    series gives numerator exactly zero, so the z-score is 0 rather than
    exploding. Worth pinning, because the same guard written as an equality
    check is a common bug."""
    clf = RegimeClassifier()
    sigs = clf.classify(DATES, CALM, CALM, FLAT_VOL)
    assert not any(s.volatility_signal for s in sigs)


def test_policy_window_covers_plus_and_minus_three_days():
    clf = RegimeClassifier(policy_event_dates=["2024-01-10"])
    assert clf._near_policy_event(pd.Timestamp("2024-01-07")) is True
    assert clf._near_policy_event(pd.Timestamp("2024-01-13")) is True
    assert clf._near_policy_event(pd.Timestamp("2024-01-06")) is False


def test_no_policy_dates_means_no_policy_signal():
    assert RegimeClassifier()._near_policy_event(pd.Timestamp("2024-01-01")) is False


# ------------------------------------------------ point-in-time volatility

def test_future_volatility_does_not_change_earlier_classifications():
    """Appending future data must not revise an earlier regime decision."""
    clf = RegimeClassifier(vol_zscore_threshold=2.0)
    dates = pd.date_range("2024-01-01", periods=25)

    # Day 21 is a spike after 20 calm observations; a larger spike arrives later.
    vol = np.full(25, 0.01)
    vol[20] = 0.05
    vol[24] = 5.0

    fx_on = np.full(25, 5.0)
    full = clf.classify(dates, fx_on, np.zeros(25), vol)
    realtime = clf.classify(dates[:21], fx_on[:21], np.zeros(21), vol[:21])

    assert realtime[20].volatility_signal is True
    assert realtime[20].label is RegimeLabel.STRESSED
    assert full[:21] == realtime


def test_later_extreme_day_does_not_suppress_earlier_volatility_spikes():
    """Each spike is evaluated against information available before its date."""
    clf = RegimeClassifier(vol_zscore_threshold=2.0)
    dates = pd.date_range("2024-01-01", periods=65)
    vol = np.full(65, 0.01)
    vol[[20, 42]] = 0.10
    vol[64] = 10.0

    sigs = clf.classify(dates, np.zeros(65), np.zeros(65), vol)
    fired = [i for i, s in enumerate(sigs) if s.volatility_signal]
    assert fired == [20, 42, 64]


def test_volatility_signal_respects_twenty_observation_cold_start():
    """The signal stays off until 20 strictly prior observations exist."""
    clf = RegimeClassifier(vol_zscore_threshold=2.0)
    for n in range(1, 21):
        vol = np.full(n, 0.01)
        vol[-1] = 10.0
        sigs = clf.classify(
            pd.date_range("2024-01-01", periods=n),
            np.zeros(n),
            np.zeros(n),
            vol,
        )
        assert not any(s.volatility_signal for s in sigs)

    vol = np.full(21, 0.01)
    vol[-1] = 10.0
    sigs = clf.classify(
        pd.date_range("2024-01-01", periods=21),
        np.zeros(21),
        np.zeros(21),
        vol,
    )
    assert sigs[-1].volatility_signal is True


# ------------------------------------------------------------------- defects


def test_policy_window_is_calendar_days_not_trading_days():
    """DEFECT: the docstring says events within +/-3 TRADING days trigger the
    signal. `_near_policy_event` differences timestamps and reads `.days`, which
    is calendar days.

    Across a weekend the two differ: a Monday event and the previous Wednesday
    are 5 calendar days apart, so the signal does not fire, though only 3
    trading days separate them. See KNOWN_ISSUES.md #7.
    """
    clf = RegimeClassifier(policy_event_dates=["2024-01-08"])   # a Monday
    wednesday = pd.Timestamp("2024-01-03")                       # 3 trading days before
    assert (pd.Timestamp("2024-01-08") - wednesday).days == 5
    assert clf._near_policy_event(wednesday) is False


def test_all_signals_are_binary_so_severity_is_invisible():
    """DEFECT, medium: every signal is a threshold crossing, so the score takes
    only 16 possible values and severity beyond the threshold is discarded.

    A basis of 2.01 and a basis of 500 produce an identical score, an identical
    label and identical notes. For a framework whose CRISIS label means
    "standard models unreliable", the inability to distinguish a marginal
    crossing from a total dislocation is a real limitation.
    See KNOWN_ISSUES.md #8.
    """
    marginal = classify_one(fx=2.01)
    catastrophic = classify_one(fx=500.0)
    assert marginal.score == catastrophic.score
    assert marginal.label is catastrophic.label
    assert marginal.notes == catastrophic.notes
