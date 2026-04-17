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
"""

from .fx_curve import EffectiveFXCurveBuilder, EffectiveFXCurve, LCSettlement
from .regime_classifier import RegimeClassifier, RegimeLabel, RegimeSignal
from .scenario_engine import ScenarioEngine, ScenarioShock, Position, DEFAULT_SCENARIOS
from .lc_priority import LCPriorityAllocator, PendingLC, AllocationPlan, LCStatus

__all__ = [
    "EffectiveFXCurveBuilder", "EffectiveFXCurve", "LCSettlement",
    "RegimeClassifier", "RegimeLabel", "RegimeSignal",
    "ScenarioEngine", "ScenarioShock", "Position", "DEFAULT_SCENARIOS",
    "LCPriorityAllocator", "PendingLC", "AllocationPlan", "LCStatus",
]
