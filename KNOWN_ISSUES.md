# Known issues

Passing tests are not evidence that every modeling assumption is correct.
The unresolved items below retain their original issue numbers. Tests that
pin unresolved behavior identify it explicitly; corrected defects have
regression tests for the intended result.

## Unresolved issues

| # | Severity | Area | Limitation and next step |
|---|----------|------|-------------------------|
| 2 | Medium | `fx_curve` | `smoothing_window` counts settlement-date rows, not days. Use a time-based window or document row-based sampling consistently. |
| 3 | Medium | `fx_curve` | Flags only positive basis dislocations. Decide whether negative dislocations belong in this importer-specific signal before changing the rule. |
| 4 | Low | `fx_curve.build` | Empty or malformed input lacks explicit boundary validation. Empty input currently fails with a pandas missing-column error. |
| 5 | Low | `fx_curve.build` | Official rates are equal-weighted while realized rates are volume-weighted. Validate official-rate uniqueness within each date. |
| 7 | Low | `regime_classifier` | Policy-event windows count calendar days although the docstring says trading days. |
| 8 | Medium | `regime_classifier` | Binary threshold signals discard severity beyond the crossing. A continuous intensity measure would require a separate calibration decision. |
| 10 | Medium | `lc_priority` | Cost efficiency is relative to the maximum cost in the current batch. Adding a request can change other scores; cross-cycle comparison needs a fixed, commodity-specific reference cost. |

The allocation policy remains greedy and whole-request, not a knapsack optimum.
It now reports `unallocated_usd` and includes remaining capacity in deferral
reasons, so unused capacity is visible rather than silently hidden.

## Corrected issues

- **#1 and #6:** FX basis smoothing and historical volatility normalization
  were corrected in earlier commits.
- **#9:** Minimum priority is checked before capacity, including well-funded
  cycles. The exact threshold remains eligible.
- **#11 and #12:** Unused capacity is reported, and invalid allocator weights
  raise `ValueError` even under `python -O`.
- **#13:** A position's settlement FX rate is used; an unset rate falls back
  to the engine baseline.
- **#14:** Fee and tariff reductions stop at zero. Baseline and shocked FX
  rates must remain positive.
- **#15:** Notional must reconcile to quantity times entry price within
  `rel_tol=1e-8` or USD 0.01 absolute tolerance. Output includes the physical
  unit and P&L per unit.
- **#16:** Scenario descriptions use absolute LCY/USD shifts and percentage
  points for additive rates. Default shocks are illustrative, not calibrated.

Regression coverage lives in `tests/test_lc_priority.py` and
`tests/test_scenario_engine.py`, including hand-calculated two-rate positions,
zero-floor fee and tariff reductions, notional reconciliation, and optimized
interpreter validation.

## Valuation boundary

The scenario engine is an additive invoice-notional sensitivity model, not a
full landed-cost repricer. For notional `N`, settlement FX `F`, FX shift `s`,
commodity return `p`, and floored rate changes `df` and `dt`, it computes:

```text
FX         = -N * s
Commodity  = -N * p  * (F + s)
LC fees    = -N * df * (F + s)
Tariffs    = -N * dt * (F + s)
Total      = FX + Commodity + LC fees + Tariffs
```

The FX/commodity cross term is included. Existing fees and tariffs are not
themselves revalued under FX or commodity shocks, and fee/tariff changes use
the original invoice notional. Consequently the total is not an exact
repricing of `N * (1 + p) * (1 + fee + tariff) * FX`. Customs bases,
compounding rules and jurisdiction-specific subsidies are outside this model.
