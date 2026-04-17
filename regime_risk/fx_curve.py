"""
fx_curve.py
-----------
Effective-FX curve builder.

In markets operating under currency controls or import restrictions,
official exchange rates published by central banks can diverge
significantly from the rates at which transactions actually clear.
Letters of Credit (LCs) may settle at materially different rates than
the official peg due to USD rationing, informal market dynamics, and
bank-level spread variations.

This module constructs an "effective FX" curve from realized LC
settlement data — the actual rates at which import transactions cleared —
rather than relying on official published rates. The resulting curve is
used downstream for accurate P&L attribution, procurement cost modeling,
and risk scenario generation.

This is especially relevant during:
  - Currency control regimes (USD rationing, LC caps)
  - Political transitions with FX policy discontinuities
  - EM financial tightening cycles
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class LCSettlement:
    """A single realized LC settlement observation."""
    settlement_date: pd.Timestamp
    official_rate: float        # Central bank published rate (LCY per USD)
    realized_rate: float        # Actual settlement rate (LCY per USD)
    lc_amount_usd: float        # Size of the LC in USD
    commodity: str              # e.g. "copper", "PVC", "energy"
    counterparty_bank: str      # Which bank cleared the LC
    settlement_days: int        # Days from LC opening to settlement


@dataclass
class EffectiveFXCurve:
    """
    Effective FX curve built from realized LC settlement data.

    Attributes
    ----------
    dates : pd.DatetimeIndex
        Observation dates.
    official_rates : np.ndarray
        Central bank official LCY/USD rates.
    effective_rates : np.ndarray
        Volume-weighted realized settlement rates.
    basis : np.ndarray
        Spread between effective and official rates (effective - official).
    regime_flags : np.ndarray
        Boolean array — True where basis exceeds the stress threshold,
        indicating the market is in a stressed / non-standard regime.
    """
    dates: pd.DatetimeIndex
    official_rates: np.ndarray
    effective_rates: np.ndarray
    basis: np.ndarray
    regime_flags: np.ndarray


class EffectiveFXCurveBuilder:
    """
    Builds effective FX curves from realized LC settlement observations.

    Parameters
    ----------
    stress_basis_threshold : float
        Basis (in LCY per USD) above which a date is flagged as stressed.
        Default 2.0 — calibrate to the specific currency pair in use.
    smoothing_window : int
        Rolling window (in days) for smoothing the effective rate.
        Default 5 trading days.
    min_observations : int
        Minimum LC settlements required to compute a reliable rate on a date.
        Dates below this threshold fall back to the official rate.
    """

    def __init__(
        self,
        stress_basis_threshold: float = 2.0,
        smoothing_window: int = 5,
        min_observations: int = 2,
    ):
        self.stress_basis_threshold = stress_basis_threshold
        self.smoothing_window = smoothing_window
        self.min_observations = min_observations

    def build(self, settlements: list[LCSettlement]) -> EffectiveFXCurve:
        """
        Build an EffectiveFXCurve from a list of LC settlement observations.

        The effective rate on each date is the volume-weighted average
        realized settlement rate across all LCs settling that day.
        On dates with fewer than min_observations, falls back to official rate.

        Parameters
        ----------
        settlements : list[LCSettlement]
            Raw LC settlement records.

        Returns
        -------
        EffectiveFXCurve
        """
        df = pd.DataFrame([
            {
                "date": s.settlement_date,
                "official_rate": s.official_rate,
                "realized_rate": s.realized_rate,
                "weight": s.lc_amount_usd,
            }
            for s in settlements
        ])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Volume-weighted average realized rate per date
        grouped = df.groupby("date").apply(
            lambda g: pd.Series({
                "official_rate": g["official_rate"].mean(),
                "effective_rate": np.average(g["realized_rate"], weights=g["weight"]),
                "n_obs": len(g),
            })
        ).reset_index()

        # Fall back to official rate on thin days
        grouped["effective_rate"] = np.where(
            grouped["n_obs"] >= self.min_observations,
            grouped["effective_rate"],
            grouped["official_rate"],
        )

        # Smooth effective rate
        grouped["effective_rate"] = (
            grouped["effective_rate"]
            .rolling(self.smoothing_window, min_periods=1)
            .mean()
        )

        grouped["basis"] = grouped["effective_rate"] - grouped["official_rate"]
        grouped["regime_flag"] = grouped["basis"] > self.stress_basis_threshold

        return EffectiveFXCurve(
            dates=pd.DatetimeIndex(grouped["date"]),
            official_rates=grouped["official_rate"].values,
            effective_rates=grouped["effective_rate"].values,
            basis=grouped["basis"].values,
            regime_flags=grouped["regime_flag"].values,
        )

    def summary(self, curve: EffectiveFXCurve) -> pd.DataFrame:
        """Return a summary DataFrame of the curve."""
        return pd.DataFrame({
            "date": curve.dates,
            "official_rate": curve.official_rates,
            "effective_rate": curve.effective_rates,
            "basis": curve.basis,
            "stressed": curve.regime_flags,
        })
