"""
regime_risk
-----------
Non-market regime risk framework for import-dependent businesses
operating in emerging market currency and policy environments.

Modules
-------
fx_curve         : Effective-FX curve builder from realized LC settlement data
regime_classifier: Rule-based regime classifier (NORMAL / ELEVATED / STRESSED / CRISIS)
scenario_engine  : Parameterized shock engine for policy and FX scenarios
lc_priority      : USD capacity allocation system for LC rationing environments
options          : Black-Scholes pricing, Greeks and Greeks-based P&L attribution
"""

from .fx_curve import EffectiveFXCurve, EffectiveFXCurveBuilder, LCSettlement
from .lc_priority import AllocationPlan, LCPriorityAllocator, LCStatus, PendingLC
from .options import Greeks, GreeksAttribution, attribute, bs_greeks, bs_price
from .regime_classifier import RegimeClassifier, RegimeLabel, RegimeSignal
from .scenario_engine import (
    DEFAULT_SCENARIOS,
    OptionPosition,
    Position,
    ScenarioEngine,
    ScenarioShock,
)

__all__ = [
    "DEFAULT_SCENARIOS",
    "AllocationPlan",
    "EffectiveFXCurve",
    "EffectiveFXCurveBuilder",
    "Greeks",
    "GreeksAttribution",
    "LCPriorityAllocator",
    "LCSettlement",
    "LCStatus",
    "OptionPosition",
    "PendingLC",
    "Position",
    "RegimeClassifier",
    "RegimeLabel",
    "RegimeSignal",
    "ScenarioEngine",
    "ScenarioShock",
    "attribute",
    "bs_greeks",
    "bs_price",
]
