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
  - Implied volatility and elapsed time, for option positions

The engine applies shocks to a position book and returns P&L attribution
by shock type, position, and commodity. Option positions are repriced with
Black-Scholes and their P&L is split into delta, gamma, vega, theta and a
residual (see options.py), alongside the FX attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, isfinite
from typing import Optional

import pandas as pd

from .options import attribute, bs_price


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
class OptionPosition:
    """
    A European option on a commodity, valued in USD with Black-Scholes.

    `quantity` is signed units of the underlying (positive long, negative
    short), for example +100 tonnes of copper calls.
    The scenario's commodity shock moves the spot, its vol shift moves the
    implied volatility, and its horizon reduces time to expiry.
    """
    position_id: str
    underlying: str             # matches a commodity_shocks / vol_shifts key
    kind: str                   # "call" or "put"
    quantity: float             # signed units of the underlying
    strike_usd: float
    expiry_years: float         # time to expiry before the scenario
    spot_usd: float             # underlying price before the scenario
    vol: float                  # implied volatility before the scenario
    rate: float = 0.0           # continuously compounded USD rate
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
      - vol_shifts: dict of {underlying: absolute implied-vol change},
        used only by option positions (0.05 = five vol points)
      - horizon_years: elapsed time over the scenario, used only by option
        positions for theta; calendar time, days/365 (1/365 = one day)
    """
    name: str
    description: str
    fx_shift: float = 0.0
    commodity_shocks: dict[str, float] = field(default_factory=dict)
    lc_fee_shift: float = 0.0
    tariff_shift: float = 0.0
    vol_shifts: dict[str, float] = field(default_factory=dict)
    horizon_years: float = 0.0


_OPTION_COLUMNS = [
    "position_id", "underlying", "kind", "quantity", "value_usd_before",
    "value_usd_after", "fx_pnl_lcy", "delta_pnl_lcy", "gamma_pnl_lcy",
    "vega_pnl_lcy", "theta_pnl_lcy", "residual_pnl_lcy", "total_pnl_lcy",
]

_OPTION_KEYS = ("delta", "gamma", "vega", "theta", "residual")


@dataclass
class ScenarioResult:
    """P&L impact of a scenario across the position book."""
    scenario_name: str
    total_pnl_lcy: float
    position_pnl: pd.DataFrame          # Per-position breakdown (procurement)
    attribution: dict[str, float]       # P&L by shock type
    option_pnl: pd.DataFrame = field(  # Per-option Greeks breakdown
        default_factory=lambda: pd.DataFrame(columns=_OPTION_COLUMNS))


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
        positions: list[Position | OptionPosition],
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

        For each option position, computes:
          1. FX P&L on the USD option value held: V0 * fx_shift
          2. Delta, gamma, vega, theta and residual P&L on the change in USD
             value, converted to LCY at the shocked FX rate

        The option parts sum exactly to the full revaluation
        V1 * F1 - V0 * F0, because V0 * (F1 - F0) + (V1 - V0) * F1 is an
        identity and the residual absorbs everything the Greeks miss.

        Parameters
        ----------
        positions : list[Position | OptionPosition]
        scenario : ScenarioShock

        Returns
        -------
        ScenarioResult
        """
        # Additive invoice-notional sensitivity, not full landed-cost repricing:
        # base fees/duties are not themselves revalued under FX/price shocks.
        shifts = [scenario.fx_shift, scenario.lc_fee_shift, scenario.tariff_shift,
                  scenario.horizon_years, *scenario.commodity_shocks.values(),
                  *scenario.vol_shifts.values()]
        if not all(isfinite(x) for x in shifts):
            raise ValueError("Scenario shifts must be finite")
        if any(x < -1 for x in scenario.commodity_shocks.values()):
            raise ValueError("Commodity shocks cannot imply negative prices")
        if scenario.horizon_years < 0:
            raise ValueError("Scenario horizon cannot be negative")
        records = []
        option_records = []

        for pos in positions:
            if isinstance(pos, OptionPosition):
                option_records.append(self._run_option(pos, scenario))
                continue
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
        odf = pd.DataFrame(option_records, columns=_OPTION_COLUMNS)
        total = df["total_pnl_lcy"].sum() + odf["total_pnl_lcy"].sum()

        attribution = {
            "fx": round(df["fx_pnl_lcy"].sum() + odf["fx_pnl_lcy"].sum(), 2),
            "commodity": round(df["commodity_pnl_lcy"].sum(), 2),
            "lc_fees": round(df["lc_pnl_lcy"].sum(), 2),
            "tariffs": round(df["tariff_pnl_lcy"].sum(), 2),
        }
        for key in _OPTION_KEYS:
            attribution[f"option_{key}"] = round(float(odf[f"{key}_pnl_lcy"].sum()), 2)

        return ScenarioResult(
            scenario_name=scenario.name,
            total_pnl_lcy=round(total, 2),
            position_pnl=df,
            attribution=attribution,
            option_pnl=odf,
        )

    def _run_option(self, pos: OptionPosition, scenario: ScenarioShock) -> dict:
        """Reprice one option under the scenario and attribute its LCY P&L."""
        base_fx = self.base_fx_rate if pos.settlement_fx is None else pos.settlement_fx
        shocked_fx = base_fx + scenario.fx_shift
        if not isfinite(base_fx) or base_fx <= 0 or not isfinite(shocked_fx) or shocked_fx <= 0:
            raise ValueError("Baseline and shocked settlement FX must be finite and positive")
        spot_after = pos.spot_usd * (1.0 + scenario.commodity_shocks.get(pos.underlying, 0.0))
        vol_after = pos.vol + scenario.vol_shifts.get(pos.underlying, 0.0)
        if vol_after <= 0:
            raise ValueError("Shocked volatility must be positive")

        a = attribute(
            S0=pos.spot_usd, S1=spot_after, vol0=pos.vol, vol1=vol_after,
            dt=scenario.horizon_years, K=pos.strike_usd, T=pos.expiry_years,
            r=pos.rate, kind=pos.kind, quantity=pos.quantity,
        )
        value_before = pos.quantity * bs_price(
            pos.spot_usd, pos.strike_usd, pos.expiry_years, pos.rate, pos.vol, pos.kind)
        value_after = value_before + a.total

        # A long option is a USD asset, so a weaker local currency is a gain.
        fx_pnl = round(value_before * scenario.fx_shift, 2)
        parts = {k: round(v * shocked_fx, 2) for k, v in a.parts().items()
                 if k != "residual"}
        total_pnl = round(value_after * shocked_fx - value_before * base_fx, 2)
        # Rounding differences go to the residual so the row reconciles exactly.
        parts["residual"] = round(total_pnl - fx_pnl - sum(parts.values()), 2)

        return {
            "position_id": pos.position_id,
            "underlying": pos.underlying,
            "kind": pos.kind,
            "quantity": pos.quantity,
            "value_usd_before": round(value_before, 2),
            "value_usd_after": round(value_after, 2),
            "fx_pnl_lcy": fx_pnl,
            **{f"{k}_pnl_lcy": parts[k] for k in _OPTION_KEYS},
            "total_pnl_lcy": total_pnl,
        }

    def run_all(
        self,
        positions: list[Position | OptionPosition],
        scenarios: Optional[list[ScenarioShock]] = None,
    ) -> pd.DataFrame:
        """
        Run all scenarios and return a summary DataFrame.

        Parameters
        ----------
        positions : list[Position | OptionPosition]
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
                "option_greeks_pnl_lcy": round(sum(
                    result.attribution[f"option_{k}"] for k in _OPTION_KEYS), 2),
            })

        return pd.DataFrame(rows).sort_values("total_pnl_lcy")
