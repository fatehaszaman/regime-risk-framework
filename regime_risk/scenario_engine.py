"""
scenario_engine.py
------------------
Scenario dashboard and shock engine for non-market regime risk.

In emerging markets with import-dependent supply chains, policy-driven
shocks — LC rationing, tariff circulars, currency controls, political
discontinuities — can materially change cost structures with little
warning. Standard market risk tools don't model these well because
they aren't price-discovery events; they're structural changes imposed
by regulators or governments.

This engine provides pre-built, parameterized shock scenarios that can
be re-run instantly against a position book when a new policy event
occurs. Scenarios are illustrative assumptions, not calibrated forecasts.

Each scenario specifies shocks to:
  - FX rates (local currency / USD effective rate shifts)
  - Commodity prices (copper, PVC, energy, etc.)
  - LC costs (opening fees, settlement spreads)
  - Tariff rates (import duty changes)

The engine applies shocks to a position book and returns P&L attribution
by shock type, position, and commodity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, isfinite
from typing import Optional

import pandas as pd


@dataclass
class Position:
    """A single open procurement position."""
    position_id: str
    commodity: str              # e.g. "copper", "PVC", "energy"
    notional_usd: float         # USD value of the position
    quantity: float             # Physical quantity (tonnes, MWh, etc.)
    unit: str                   # "tonne", "MWh", etc.
    entry_price_usd: float      # Price per unit at entry
    lc_fee_pct: float           # LC opening fee as % of notional
    tariff_rate: float          # Import tariff rate (e.g. 0.15 = 15%)
    settlement_fx: Optional[float] = None  # LCY/USD; None uses engine baseline


@dataclass
class ScenarioShock:
    """
    A parameterized shock applied to a position book.

    All shocks are additive unless noted:
      - fx_shift: LCY/USD shift (positive = local currency weakens)
      - commodity_shocks: dict of {commodity: price_change_pct}
      - lc_fee_shift: absolute shift in LC fee percentage
      - tariff_shift: absolute shift in tariff rate
    """
    name: str
    description: str
    fx_shift: float = 0.0
    commodity_shocks: dict[str, float] = field(default_factory=dict)
    lc_fee_shift: float = 0.0
    tariff_shift: float = 0.0


@dataclass
class ScenarioResult:
    """P&L impact of a scenario across the position book."""
    scenario_name: str
    total_pnl_lcy: float
    position_pnl: pd.DataFrame          # Per-position breakdown
    attribution: dict[str, float]       # P&L by shock type


# Pre-built scenarios covering common emerging market regime events.
# Illustrative parameters, not an empirically calibrated crisis distribution.
DEFAULT_SCENARIOS = [
    ScenarioShock(
        name="LC_Rationing_Onset",
        description="Central bank restricts USD LC allocations; opening fees spike 150bps, mild FX weakening",
        fx_shift=1.5,
        lc_fee_shift=0.015,
    ),
    ScenarioShock(
        name="Political_Discontinuity",
        description="Government transition: LCY/USD +8.5, LC fees +2 percentage points, tariffs +5 percentage points",
        fx_shift=8.5,
        commodity_shocks={"copper": 0.03, "PVC": 0.02},
        lc_fee_shift=0.02,
        tariff_shift=0.05,
    ),
    ScenarioShock(
        name="Commodity_FX_Correlation_Shock",
        description="Correlated move: key commodity +15%, LCY/USD +5 — import cost double-hit",
        fx_shift=5.0,
        commodity_shocks={"copper": 0.15},
    ),
    ScenarioShock(
        name="Tariff_Circular",
        description="New import tariff circular increases duties by 10 percentage points",
        tariff_shift=0.10,
    ),
    ScenarioShock(
        name="Severe_Stress",
        description="Combined tail scenario: LCY/USD +15, commodities +8-20%, LC fees +300bps, tariffs +15 percentage points",
        fx_shift=15.0,
        commodity_shocks={"copper": 0.20, "PVC": 0.10, "energy": 0.08},
        lc_fee_shift=0.03,
        tariff_shift=0.15,
    ),
]


class ScenarioEngine:
    """
    Applies parameterized shocks to a position book and computes P&L impact.

    Parameters
    ----------
    base_fx_rate : float
        Current effective LCY/USD rate (baseline before shocks).
    """

    def __init__(self, base_fx_rate: float = 110.0):
        if not isfinite(base_fx_rate) or base_fx_rate <= 0:
            raise ValueError("Base FX rate must be finite and positive")
        self.base_fx_rate = base_fx_rate

    def run(
        self,
        positions: list[Position],
        scenario: ScenarioShock,
    ) -> ScenarioResult:
        """
        Apply a scenario shock to the position book.

        For each position, computes:
          1. FX P&L: change in LCY cost from FX shift
          2. Commodity P&L: change in USD cost from commodity price shift,
             converted to LCY at shocked FX rate
          3. LC cost P&L: change in LC fees
          4. Tariff P&L: change in import duty costs

        Parameters
        ----------
        positions : list[Position]
        scenario : ScenarioShock

        Returns
        -------
        ScenarioResult
        """
        # Additive invoice-notional sensitivity, not full landed-cost repricing:
        # base fees/duties are not themselves revalued under FX/price shocks.
        shifts = [scenario.fx_shift, scenario.lc_fee_shift, scenario.tariff_shift,
                  *scenario.commodity_shocks.values()]
        if not all(isfinite(x) for x in shifts):
            raise ValueError("Scenario shifts must be finite")
        if any(x < -1 for x in scenario.commodity_shocks.values()):
            raise ValueError("Commodity shocks cannot imply negative prices")
        records = []

        for pos in positions:
            values = (pos.notional_usd, pos.quantity, pos.entry_price_usd)
            if any(not isfinite(x) or x <= 0 for x in values):
                raise ValueError("Notional, quantity and entry price must be finite and positive")
            if not isclose(pos.notional_usd, pos.quantity * pos.entry_price_usd,
                           rel_tol=1e-8, abs_tol=0.01):
                raise ValueError("Notional must equal quantity times entry price")
            if any(not isfinite(x) or x < 0 for x in (pos.lc_fee_pct, pos.tariff_rate)):
                raise ValueError("Baseline fees and tariffs must be finite and non-negative")
            base_fx = self.base_fx_rate if pos.settlement_fx is None else pos.settlement_fx
            shocked_fx = base_fx + scenario.fx_shift
            if not isfinite(base_fx) or base_fx <= 0 or not isfinite(shocked_fx) or shocked_fx <= 0:
                raise ValueError("Baseline and shocked settlement FX must be finite and positive")
            # FX P&L: same USD notional costs more LCY when currency weakens
            fx_pnl_lcy = -pos.notional_usd * scenario.fx_shift

            # Commodity P&L: price change on the position
            commodity_shock_pct = scenario.commodity_shocks.get(pos.commodity, 0.0)
            commodity_pnl_usd = -pos.notional_usd * commodity_shock_pct
            commodity_pnl_lcy = commodity_pnl_usd * shocked_fx

            # LC fee P&L
            lc_delta = max(0.0, pos.lc_fee_pct + scenario.lc_fee_shift) - pos.lc_fee_pct
            lc_pnl_lcy = -pos.notional_usd * lc_delta * shocked_fx

            # Tariff P&L
            tariff_delta = max(0.0, pos.tariff_rate + scenario.tariff_shift) - pos.tariff_rate
            tariff_pnl_lcy = -pos.notional_usd * tariff_delta * shocked_fx

            total_pnl_lcy = fx_pnl_lcy + commodity_pnl_lcy + lc_pnl_lcy + tariff_pnl_lcy

            records.append({
                "position_id": pos.position_id,
                "commodity": pos.commodity,
                "notional_usd": pos.notional_usd,
                "unit": pos.unit,
                "pnl_per_unit_lcy": round(total_pnl_lcy / pos.quantity, 2),
                "fx_pnl_lcy": round(fx_pnl_lcy, 2),
                "commodity_pnl_lcy": round(commodity_pnl_lcy, 2),
                "lc_pnl_lcy": round(lc_pnl_lcy, 2),
                "tariff_pnl_lcy": round(tariff_pnl_lcy, 2),
                "total_pnl_lcy": round(total_pnl_lcy, 2),
            })

        df = pd.DataFrame(records, columns=[
            "position_id", "commodity", "notional_usd", "unit", "pnl_per_unit_lcy",
            "fx_pnl_lcy", "commodity_pnl_lcy", "lc_pnl_lcy", "tariff_pnl_lcy", "total_pnl_lcy",
        ])
        total = df["total_pnl_lcy"].sum()

        attribution = {
            "fx": round(df["fx_pnl_lcy"].sum(), 2),
            "commodity": round(df["commodity_pnl_lcy"].sum(), 2),
            "lc_fees": round(df["lc_pnl_lcy"].sum(), 2),
            "tariffs": round(df["tariff_pnl_lcy"].sum(), 2),
        }

        return ScenarioResult(
            scenario_name=scenario.name,
            total_pnl_lcy=round(total, 2),
            position_pnl=df,
            attribution=attribution,
        )

    def run_all(
        self,
        positions: list[Position],
        scenarios: Optional[list[ScenarioShock]] = None,
    ) -> pd.DataFrame:
        """
        Run all scenarios and return a summary DataFrame.

        Parameters
        ----------
        positions : list[Position]
        scenarios : list[ScenarioShock], optional
            Defaults to DEFAULT_SCENARIOS.
        """
        if scenarios is None:
            scenarios = DEFAULT_SCENARIOS

        rows = []
        for scenario in scenarios:
            result = self.run(positions, scenario)
            rows.append({
                "scenario": result.scenario_name,
                "total_pnl_lcy": result.total_pnl_lcy,
                "fx_pnl_lcy": result.attribution["fx"],
                "commodity_pnl_lcy": result.attribution["commodity"],
                "lc_pnl_lcy": result.attribution["lc_fees"],
                "tariff_pnl_lcy": result.attribution["tariffs"],
            })

        return pd.DataFrame(rows).sort_values("total_pnl_lcy")
