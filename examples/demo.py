"""
demo.py
-------
End-to-end demonstration of the regime_risk framework.

Simulates a 6-month window where a currency control regime gradually
tightens, culminating in a policy discontinuity event, then shows
how each module responds.
"""

import numpy as np
import pandas as pd

from regime_risk import (
    EffectiveFXCurveBuilder, LCSettlement,
    RegimeClassifier,
    ScenarioEngine, Position, DEFAULT_SCENARIOS,
    LCPriorityAllocator, PendingLC,
)

np.random.seed(42)

# ── 1. Generate synthetic LC settlement data ─────────────────────────────────

print("=" * 60)
print("1. EFFECTIVE FX CURVE")
print("=" * 60)

dates = pd.date_range("2024-02-01", "2024-07-31", freq="B")
n = len(dates)

# Official rate drifts from 108 to 114; realized rate diverges from June onward
official = np.linspace(108, 114, n) + np.random.normal(0, 0.2, n)
split = int(n * 0.6)
divergence = np.concatenate([np.zeros(split), np.linspace(0, 9, n - split)])
realized = official + divergence + np.random.normal(0, 0.3, n)

settlements = []
commodities = ["copper", "PVC", "energy"]
banks = ["Bank_A", "Bank_B", "Bank_C"]

for i, date in enumerate(dates):
    for _ in range(np.random.randint(1, 4)):
        settlements.append(LCSettlement(
            settlement_date=date,
            official_rate=round(official[i], 4),
            realized_rate=round(realized[i] + np.random.normal(0, 0.15), 4),
            lc_amount_usd=np.random.uniform(50_000, 500_000),
            commodity=np.random.choice(commodities),
            counterparty_bank=np.random.choice(banks),
            settlement_days=np.random.randint(30, 90),
        ))

builder = EffectiveFXCurveBuilder(stress_basis_threshold=2.0, smoothing_window=5)
curve = builder.build(settlements)
summary = builder.summary(curve)

stressed_days = summary["stressed"].sum()
max_basis = summary["basis"].max()
print(f"Date range     : {summary['date'].min().date()} to {summary['date'].max().date()}")
print(f"Stressed days  : {stressed_days} / {len(summary)}")
print(f"Max basis      : {max_basis:.2f} LCY/USD")
print(f"Final eff. rate: {summary['effective_rate'].iloc[-1]:.2f} vs official {summary['official_rate'].iloc[-1]:.2f}")
print()

# ── 2. Regime classification ──────────────────────────────────────────────────

print("=" * 60)
print("2. REGIME CLASSIFIER")
print("=" * 60)

lc_utilization = np.clip(
    np.linspace(0.60, 0.95, n) + np.random.normal(0, 0.03, n), 0, 1
)
realized_vol = np.abs(np.random.normal(0.008, 0.003, n))
realized_vol[int(n * 0.65):int(n * 0.70)] *= 3.5   # vol spike at policy event

classifier = RegimeClassifier(
    fx_basis_threshold=2.0,
    lc_utilization_threshold=0.85,
    vol_zscore_threshold=2.0,
    policy_event_dates=["2024-06-15", "2024-07-01"],
)

signals = classifier.classify(
    dates=curve.dates,
    fx_basis=curve.basis,
    lc_utilization=lc_utilization[:len(curve.dates)],
    realized_vol=realized_vol[:len(curve.dates)],
)

regime_df = classifier.to_dataframe(signals)
print(regime_df["regime"].value_counts().to_string())
print()
print("Last 5 observations:")
print(regime_df[["date", "regime", "score", "notes"]].tail(5).to_string(index=False))
print()

# ── 3. Scenario engine ────────────────────────────────────────────────────────

print("=" * 60)
print("3. SCENARIO DASHBOARD")
print("=" * 60)

positions = [
    Position("POS-001", "copper", 1_200_000, 150, "tonne", 8000, 0.015, 0.12, 110.0),
    Position("POS-002", "PVC",    800_000,   400, "tonne", 2000, 0.012, 0.10, 110.0),
    Position("POS-003", "energy", 500_000,  5000, "MWh",   100,  0.010, 0.08, 110.0),
]

engine = ScenarioEngine(base_fx_rate=110.0)
results = engine.run_all(positions)

print(results[["scenario", "total_pnl_lcy", "fx_pnl_lcy", "commodity_pnl_lcy",
               "lc_pnl_lcy", "tariff_pnl_lcy"]].to_string(index=False))
print()

# ── 4. LC priority allocation ─────────────────────────────────────────────────

print("=" * 60)
print("4. LC PRIORITY ALLOCATION")
print("=" * 60)

pending = [
    PendingLC("LC-A", "copper", 450_000, urgency_score=9, strategic_score=8,
              days_until_expiry=20, unit_cost_usd=8100, quantity=55, unit="tonne",
              notes="Critical production input — line stops without this"),
    PendingLC("LC-B", "PVC",    300_000, urgency_score=6, strategic_score=7,
              days_until_expiry=45, unit_cost_usd=2050, quantity=145, unit="tonne"),
    PendingLC("LC-C", "energy", 200_000, urgency_score=5, strategic_score=5,
              days_until_expiry=60, unit_cost_usd=105,  quantity=1900, unit="MWh"),
    PendingLC("LC-D", "copper", 350_000, urgency_score=4, strategic_score=3,
              days_until_expiry=90, unit_cost_usd=8300, quantity=42, unit="tonne",
              notes="Optional buffer stock"),
    PendingLC("LC-E", "PVC",    150_000, urgency_score=2, strategic_score=2,
              days_until_expiry=120, unit_cost_usd=2200, quantity=68, unit="tonne"),
]

allocator = LCPriorityAllocator()
plan = allocator.allocate(pending, available_usd=800_000)

print(f"Capacity      : ${plan.total_capacity_usd:,.0f}")
print(f"Allocated     : ${plan.total_allocated_usd:,.0f} ({plan.utilization_rate:.1%})")
print(f"Deferred      : ${plan.total_deferred_usd:,.0f}")
print(f"Cancelled     : ${plan.total_cancelled_usd:,.0f}")
print()
print(plan.to_dataframe()[["lc_id", "commodity", "amount_usd", "status",
                            "priority_score", "reason"]].to_string(index=False))
