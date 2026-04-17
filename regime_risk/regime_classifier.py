"""
regime_classifier.py
--------------------
Regime classifier for non-market structural risk.

Standard volatility-regime classifiers (e.g. HMM, GARCH-based) model
market regimes — periods of elevated vol, correlation breakdown, etc.
This classifier is designed for a different problem: detecting when
*non-market* structural conditions have shifted in ways that invalidate
standard pricing and risk assumptions.

Relevant regime shift types:
  - LC rationing onset (USD allocation constrained by central bank)
  - Official FX rate decoupling from realized settlement rates
  - Government transitions and policy discontinuities
  - Tariff / import circular issuance creating abrupt cost structure changes

The classifier uses a rule-based scoring approach across four signals,
producing a regime label and a confidence score at each timestep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RegimeLabel(str, Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"        # One or more signals flashing, not yet critical
    STRESSED = "stressed"        # Multiple signals, meaningful operational impact
    CRISIS = "crisis"            # Full regime collapse — standard models unreliable


@dataclass
class RegimeSignal:
    """A single timestep regime assessment."""
    date: pd.Timestamp
    label: RegimeLabel
    score: float                        # 0.0 (normal) to 1.0 (crisis)
    fx_basis_signal: bool               # Effective vs official FX spread > threshold
    lc_utilization_signal: bool         # LC utilization rate above stress threshold
    volatility_signal: bool             # Realized vol spike
    policy_event_signal: bool           # Known policy discontinuity on or near date
    notes: str = ""


class RegimeClassifier:
    """
    Classifies market and operational regime at each timestep.

    Scoring logic
    -------------
    Each of four signals contributes a weight to a total score:
      - FX basis signal:        0.35  (most informative in currency control regimes)
      - LC utilization signal:  0.30
      - Policy event signal:    0.20
      - Volatility signal:      0.15

    Score thresholds → RegimeLabel:
      [0.00, 0.20) → NORMAL
      [0.20, 0.45) → ELEVATED
      [0.45, 0.70) → STRESSED
      [0.70, 1.00] → CRISIS

    Parameters
    ----------
    fx_basis_threshold : float
        Basis (LCY/USD) above which FX signal fires. Default 2.0.
    lc_utilization_threshold : float
        LC utilization rate (0–1) above which LC signal fires. Default 0.85.
    vol_zscore_threshold : float
        Z-score of realized vol above which vol signal fires. Default 2.0.
    policy_event_dates : list[str]
        Known dates of policy discontinuities (e.g. new circulars, regime change).
        Events within ±3 trading days of a date trigger the policy signal.
    """

    SIGNAL_WEIGHTS = {
        "fx_basis": 0.35,
        "lc_utilization": 0.30,
        "volatility": 0.15,
        "policy_event": 0.20,
    }

    THRESHOLDS = [
        (0.70, RegimeLabel.CRISIS),
        (0.45, RegimeLabel.STRESSED),
        (0.20, RegimeLabel.ELEVATED),
        (0.00, RegimeLabel.NORMAL),
    ]

    def __init__(
        self,
        fx_basis_threshold: float = 2.0,
        lc_utilization_threshold: float = 0.85,
        vol_zscore_threshold: float = 2.0,
        policy_event_dates: Optional[list[str]] = None,
    ):
        self.fx_basis_threshold = fx_basis_threshold
        self.lc_utilization_threshold = lc_utilization_threshold
        self.vol_zscore_threshold = vol_zscore_threshold
        self.policy_event_dates = pd.to_datetime(policy_event_dates or [])

    def _score(self, fx_b: bool, lc_u: bool, vol: bool, policy: bool) -> float:
        w = self.SIGNAL_WEIGHTS
        return (
            w["fx_basis"] * fx_b
            + w["lc_utilization"] * lc_u
            + w["volatility"] * vol
            + w["policy_event"] * policy
        )

    def _label(self, score: float) -> RegimeLabel:
        for threshold, label in self.THRESHOLDS:
            if score >= threshold:
                return label
        return RegimeLabel.NORMAL

    def _near_policy_event(self, date: pd.Timestamp, window: int = 3) -> bool:
        if len(self.policy_event_dates) == 0:
            return False
        delta = np.abs((self.policy_event_dates - date).days)
        return bool(delta.min() <= window)

    def classify(
        self,
        dates: pd.DatetimeIndex,
        fx_basis: np.ndarray,
        lc_utilization: np.ndarray,
        realized_vol: np.ndarray,
    ) -> list[RegimeSignal]:
        """
        Classify regime at each timestep.

        Parameters
        ----------
        dates : pd.DatetimeIndex
        fx_basis : np.ndarray
            Daily basis between effective and official FX rate (LCY/USD).
        lc_utilization : np.ndarray
            Daily LC utilization rate (fraction of allocated USD capacity used).
        realized_vol : np.ndarray
            Daily realized volatility of the relevant FX or commodity series.

        Returns
        -------
        list[RegimeSignal]
        """
        vol_mean = np.mean(realized_vol)
        vol_std = np.std(realized_vol) + 1e-10
        vol_zscore = (realized_vol - vol_mean) / vol_std

        signals = []
        for i, date in enumerate(dates):
            fx_signal = bool(fx_basis[i] > self.fx_basis_threshold)
            lc_signal = bool(lc_utilization[i] > self.lc_utilization_threshold)
            vol_signal = bool(vol_zscore[i] > self.vol_zscore_threshold)
            policy_signal = self._near_policy_event(date)

            score = self._score(fx_signal, lc_signal, vol_signal, policy_signal)
            label = self._label(score)

            active = [
                name for name, fired in [
                    ("FX basis", fx_signal),
                    ("LC utilization", lc_signal),
                    ("vol spike", vol_signal),
                    ("policy event", policy_signal),
                ] if fired
            ]
            notes = f"Active signals: {', '.join(active)}" if active else "No signals"

            signals.append(RegimeSignal(
                date=date,
                label=label,
                score=round(score, 4),
                fx_basis_signal=fx_signal,
                lc_utilization_signal=lc_signal,
                volatility_signal=vol_signal,
                policy_event_signal=policy_signal,
                notes=notes,
            ))

        return signals

    def to_dataframe(self, signals: list[RegimeSignal]) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "date": s.date,
                "regime": s.label.value,
                "score": s.score,
                "fx_basis_signal": s.fx_basis_signal,
                "lc_utilization_signal": s.lc_utilization_signal,
                "volatility_signal": s.volatility_signal,
                "policy_event_signal": s.policy_event_signal,
                "notes": s.notes,
            }
            for s in signals
        ])
