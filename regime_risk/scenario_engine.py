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
occurs — cutting response time from days to minutes.

Each scenario specifies shocks to:
  - FX rates (local currency / USD effective rate shifts)
  - Commodity prices (copper, PVC, energy, etc.)
  - LC costs (opening fees, settlement spreads)
  - Tariff rates (import duty changes)

The engine applies shocks to a position book and returns P&L attribution
by shock type, position, and commodity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


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
    settlement_fx: float        # Expected LCY/USD rate at settlement


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
# Calibrated to historical observations across EM currency crises,
# import control regimes, and political discontinuities.
DEFAULT_SCENARIOS = [
    ScenarioShock(
        name="LC_Rationing_Onset",
        description="Central bank restricts USD LC allocations; opening fees spike 150bps, mild FX weakening",
        fx_shift=1.5,
        lc_fee_shift=0.015,
    ),
    ScenarioShock(
        name="Political_Discontinuity",
        description="Government transition — local currency devalues ~8%, tariff policy uncertainty",
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
        description="New import tariff circular increases duties by 10%",
        tariff_shift=0.10,
    ),
    ScenarioShock(
        name="Severe_Stress",
        description="Combined tail scenario: LCY -15%, commodities +10-20%, LC fees +300bps, tariffs +15%",
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
        shocked_fx = self.base_fx_rate + scenario.fx_shift
        records = []

        for pos in positions:
            # FX P&L: same USD notional costs more LCY when currency weakens
            fx_pnl_lcy = -pos.notional_usd * scenario.fx_shift

            # Commodity P&L: price change on the position
            commodity_shock_pct = scenario.commodity_shocks.get(pos.commodity, 0.0)
            commodity_pnl_usd = -pos.notional_usd * commodity_shock_pct
            commodity_pnl_lcy = commodity_pnl_usd * shocked_fx

            # LC fee P&L
            lc_pnl_lcy = -pos.notional_usd * scenario.lc_fee_shift * shocked_fx

            # Tariff P&L
            tariff_pnl_lcy = -pos.notional_usd * scenario.tariff_shift * shocked_fx

            total_pnl_lcy = fx_pnl_lcy + commodity_pnl_lcy + lc_pnl_lcy + tariff_pnl_lcy

            records.append({
                "position_id": pos.position_id,
                "commodity": pos.commodity,
                "notional_usd": pos.notional_usd,
                "fx_pnl_lcy": round(fx_pnl_lcy, 2),
                "commodity_pnl_lcy": round(commodity_pnl_lcy, 2),
                "lc_pnl_lcy": round(lc_pnl_lcy, 2),
                "tariff_pnl_lcy": round(tariff_pnl_lcy, 2),
                "total_pnl_lcy": round(total_pnl_lcy, 2),
            })

        df = pd.DataFrame(records)
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
