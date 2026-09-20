# Known issues

Found by writing the test suite. Each item has a test that pins the current
behaviour and cites this file by number.

Convention: a test that asserts an unresolved defect says so in its docstring.
When the bug is fixed, that test is replaced with a regression test for the
corrected result and the issue is removed from this list.

Only unresolved issues are listed here. Issues #1 and #6 were replaced with
regression tests after the FX curve became trend-invariant and the volatility
signal became point-in-time safe.

| # | Severity | Area | Issue |
|---|----------|------|-------|
| 2 | Medium | `fx_curve` | `smoothing_window` counts settlement rows, not the documented days |
| 3 | Medium | `fx_curve` | `regime_flags` is one-sided; a large negative basis is never flagged |
| 4 | Low | `fx_curve.build` | No input validation; empty input fails on a missing column |
| 5 | Low | `fx_curve.build` | Official rate equal-weighted while effective rate is volume-weighted |
| 7 | Low | `regime_classifier` | Policy window is calendar days, documented as trading days |
| 8 | Medium | `regime_classifier` | All signals binary, so severity beyond a threshold is discarded |
| 9 | High | `lc_priority.allocate` | `min_priority_threshold` is ignored whenever capacity is ample |
| 10 | Medium | `lc_priority` | Cost efficiency normalised within the batch, so scores are not stable |
| 11 | Medium | `lc_priority.allocate` | Greedy first-fit leaves capacity unused without reporting slack |
| 12 | Low | `lc_priority` | Weight validation uses `assert`, stripped under `python -O` |
| 13 | High | `scenario_engine` | `Position.settlement_fx` is never used; all positions valued at one rate |
| 14 | High | `scenario_engine` | Shocks are not clamped against position levels; tariffs can go negative |
| 15 | Medium | `scenario_engine` | `entry_price_usd`, `quantity` and `unit` are never used |
| 16 | Low | `scenario_engine` | Scenario descriptions quote percentages that hold only at the default FX rate |

---

## fx_curve

### 2. `smoothing_window` counts rows, not days (Medium)

Documented as "Rolling window (in days)". `.rolling(n)` counts rows of the
grouped frame, and those rows are settlement dates. LC settlement data is sparse
and irregular, so five rows can span months; the tests include five dates spread
over 100 calendar days being averaged as though adjacent.

Fix: reindex to a daily (or business-daily) index before rolling, or use
`.rolling("5D", on="date")` and document which.

Test: `test_smoothing_window_counts_rows_not_calendar_days`.

### 3. `regime_flags` is one-sided (Medium)

`basis > stress_basis_threshold` never fires on a large negative basis, where
LCs settle far *below* the official rate. Under import controls the basis is
usually positive, which makes the one-sided test defensible, but a negative
dislocation remains invisible unless that directional choice is made explicit.

Fix: flag on `abs(basis)`, or keep the one-sided test and say so explicitly in
the docstring and the README.

Test: `test_negative_basis_is_never_flagged_however_large`.

### 4. No input validation (Low)

`build([])` raises `KeyError: 'date'` from inside pandas. A thin or malformed
settlement list fails somewhere unhelpful rather than at the boundary.

Fix: raise `ValueError` on empty input and validate that rates are positive and
`lc_amount_usd` is non-negative, since a zero or negative weight silently
corrupts `np.average`.

Test: `test_empty_settlement_list_raises_an_opaque_error`.

### 5. Official rate equal-weighted, effective rate volume-weighted (Low)

Within a date the official rate is aggregated with `.mean()`. The official rate
should be identical across settlements on a given day, so any disagreement means
bad input — and it is silently averaged rather than reported.

Fix: assert uniqueness per date and raise on disagreement.

Test: `test_official_rate_is_equal_weighted_while_effective_is_volume_weighted`.

## regime_classifier

### 7. Policy window is calendar days (Low)

The docstring says events within ±3 **trading** days trigger the signal.
`_near_policy_event` differences timestamps and reads `.days`, which is calendar
days. Across a weekend the two diverge: a Monday event and the preceding
Wednesday are 3 trading days apart and 5 calendar days apart, so the signal does
not fire.

Fix: count with a business-day offset, or reword to "calendar days".

Test: `test_policy_window_is_calendar_days_not_trading_days`.

### 8. Binary signals discard severity (Medium)

Every signal is a threshold crossing, so the score takes only 16 possible
values. A basis of 2.01 and a basis of 500 yield an identical score, label and
notes. For a framework whose CRISIS label is documented as "standard models
unreliable", not distinguishing a marginal crossing from a total dislocation is
a real limitation rather than a simplification.

Fix: keep the binary signals for interpretability but add a continuous intensity
per signal (e.g. `basis / threshold`, capped), and report both.

Test: `test_all_signals_are_binary_so_severity_is_invisible`.

## lc_priority

### 9. `min_priority_threshold` is ignored when capacity is ample (High)

The threshold is only consulted in the `elif` branch of the allocation loop,
which is reached only after capacity has run out:

```python
if remaining >= lc.amount_usd:        # approve, no score check
    ...
elif score < self.min_priority_threshold:   # cancel
```

An LC scoring far below the threshold is APPROVED whenever there happens to be
room. The parameter is documented as "LCs below this score are cancelled rather
than deferred", and it silently does nothing in exactly the case where the
control matters — a well-funded cycle, where it is supposed to stop capacity
going to requests that do not justify it. The test approves an LC scoring 3.0
against a threshold of 20.0.

Fix: check the threshold before the capacity test, so a sub-threshold LC is
cancelled regardless of available capacity.

Test: `test_minimum_priority_threshold_is_ignored_when_capacity_is_ample`.

### 10. Scores are not stable across batches (Medium)

`_cost_efficiency` normalises by the most expensive LC *in the batch*, so an
LC's score changes when unrelated LCs are added or removed. The tests show the
same request scoring 85.0 alone and 98.5 alongside a dearer one.

Decisions are therefore not reproducible across cycles, and an approval cannot
be explained by reference to the LC alone — which matters if anyone has to
justify why a shipment was cancelled.

Fix: normalise against a fixed reference cost per commodity, configured rather
than inferred, so the score is a property of the LC.

Test: `test_a_score_depends_on_which_other_lcs_are_in_the_batch`.

### 11. Greedy first-fit leaves capacity unused (Medium)

Allocation walks the priority ranking and funds anything that fits, so a large
high-priority LC that does not fit is deferred while smaller lower-priority LCs
are funded from the remainder. That is a defensible policy, and the real issue
is that the resulting slack is not reported — `utilization_rate` is the only
clue, and nothing says how much was left or why.

This is flagged rather than fixed because the alternative is a knapsack solve,
which is a design decision rather than a bug fix.

Fix, minimum: report `unallocated_usd` on the plan and note in each deferral how
much capacity remained.

Test: `test_capacity_is_left_unused_because_allocation_is_strictly_greedy`.

### 12. Weight validation uses `assert` (Low)

The weights-sum-to-one check is an `assert`, which Python removes entirely under
`-O`. Input validation that disappears in an optimised interpreter is not input
validation; with it stripped, weights summing to 2.0 produce scores up to 200 on
a scale documented as 0–100. The test constructs exactly that in a `-O`
subprocess.

Fix: `raise ValueError`.

Test: `test_weight_validation_uses_assert_and_vanishes_under_optimisation`.

## scenario_engine

### 13. `Position.settlement_fx` is never used (High)

Documented as "Expected LCY/USD rate at settlement" and never read. Every
position is revalued at the single engine-level `base_fx_rate`, so a book whose
LCs were opened over months of a moving rate is treated as settling at one rate.
Positions with settlement rates of 80 and 200 produce identical P&L.

The field reads as load-bearing and is inert, which is worse than its absence
would be.

Fix: use `pos.settlement_fx` as each position's baseline and fall back to
`base_fx_rate` only when it is unset.

Test: `test_per_position_settlement_fx_is_never_used`.

### 14. Shocks are not clamped against position levels (High)

The engine applies shifts without reference to the levels carried on the
`Position`, so nothing prevents a shock implying a negative tariff or a negative
LC fee. A position at a 5% duty given `tariff_shift = -0.20` reaches an implied
−15%, and the engine books a 22,000,000 LCY **gain** — as though customs paid
the importer a subsidy.

`Position.tariff_rate` and `Position.lc_fee_pct` already exist, so the engine
has what it needs to clamp and does not use it.

Fix: compute the shocked level as `max(0.0, level + shift)` per position and
derive P&L from the change in level.

Test: `test_shocks_are_not_clamped_so_tariffs_can_go_negative`.

### 15. `entry_price_usd`, `quantity` and `unit` are never used (Medium)

The engine works purely in notional and percentage shifts. It therefore cannot
report cost per tonne — the unit a procurement desk actually works in — and it
never catches a position whose notional disagrees with
`quantity * entry_price_usd`. The test passes a position off by 9x and gets a
clean answer.

Fix: validate the identity on construction within a tolerance, and add per-unit
cost to the output.

Test: `test_entry_price_and_quantity_are_never_used`.

### 16. Scenario descriptions quote rate-dependent percentages (Low)

`fx_shift` is an absolute LCY/USD move while the descriptions express the same
thing as a percentage devaluation. The two agree only at `base_fx_rate = 110`, a
configurable default. `Political_Discontinuity` says the currency "devalues ~8%"
with `fx_shift = 8.5`: 7.7% at 110, 17% at 50. Meanwhile the reported FX P&L is
`notional * shift` and does not depend on the base rate at all — so the
description is rate-dependent and the number it describes is not.

Separately, `Tariff_Circular` is described as increasing duties "by 10%" and
implemented as 10 percentage *points*, which on a 5% duty is a 200% relative
increase.

Fix: state shifts in the units the parameters actually use, and either express
FX shocks as a percentage of the base rate or drop the percentages from the
descriptions.

Tests: `test_scenario_descriptions_quote_percentages_only_true_at_the_default_rate`,
`test_tariff_circular_description_confuses_points_with_percent`.

---

## A note on what CI was checking before

The previous workflow ran an import, the demo, and this step:

```yaml
- name: Run tests if present
  run: |
    if [ -d tests ] || ls test_*.py 2>/dev/null; then
      pip install pytest && pytest -q
    else
      echo "No test suite found; skipping pytest."
    fi
```

There were no tests, so that step printed a message and passed. The badge was
green on a repository in which none of the sixteen issues above would have been
caught. The conditional has been removed — CI now runs the suite
unconditionally, so it fails if the tests are deleted.
