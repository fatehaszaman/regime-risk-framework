# Regime Risk Framework

[![Python CI](https://github.com/fatehaszaman/regime-risk-framework/actions/workflows/python-ci.yml/badge.svg)](https://github.com/fatehaszaman/regime-risk-framework/actions/workflows/python-ci.yml)

A Python framework for modeling non-market regime risk in import-dependent businesses operating in emerging market currency and policy environments.

Standard risk tools model market risk: vol spikes, correlation breakdowns, fat tails. This framework models a different problem: what happens when the structural rules change. Currency controls, LC rationing, tariff circulars, and political discontinuities don't show up in price history until it's too late. This framework makes them first-class risk inputs.

## The Problem

In markets operating under currency controls or import restrictions, businesses face risks that standard VaR and scenario tools miss:

- Official FX rates decouple from realized settlement rates. Central bank published rates become unreliable; actual transaction costs diverge significantly.
- USD capacity gets rationed. Businesses can't open all the LCs they need and have to prioritize.
- Policy discontinuities arrive suddenly. A new tariff circular or import restriction changes the cost structure overnight.
- Regime shifts invalidate model assumptions. A model calibrated on normal-market data gives wrong answers in a crisis.

## Modules

### `fx_curve.py` -- Effective FX Curve Builder

Constructs an effective FX curve from realized LC settlement data rather than official published rates. During currency control regimes, the gap between official and effective rates is itself a risk signal.

- Volume-weighted average realized settlement rates per date
- Configurable fallback to official rates on thin observation days
- Rolling smoothing with adjustable window
- Regime flagging when basis exceeds a stress threshold

The smoothing and the basis calculation currently interact badly: the effective
rate is smoothed and then differenced against the unsmoothed official rate,
which turns a trend in the level into a basis that is not there. This is the
first entry in [KNOWN_ISSUES.md](KNOWN_ISSUES.md) and it matters, because the
basis is the largest-weighted input to the classifier below.

### `regime_classifier.py` -- Regime Classifier

Rule-based classifier that assigns a regime label (NORMAL / ELEVATED / STRESSED / CRISIS) at each timestep based on four signals:

| Signal | Weight | Description |
|---|---|---|
| FX basis | 35% | Spread between effective and official FX rate |
| LC utilization | 30% | Fraction of USD allocation capacity in use |
| Policy event | 20% | Proximity to known policy discontinuity dates |
| Volatility | 15% | Z-score of realized vol |

The volatility z-score is currently computed from full-sample statistics, so a
label depends on data from after its own date and cannot be reproduced in real
time. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) #6.

### `scenario_engine.py` -- Scenario Dashboard

Applies parameterized shocks to a position book and returns P&L attribution by shock type. Pre-built scenarios:

| Scenario | Description |
|---|---|
| `LC_Rationing_Onset` | Central bank restricts USD allocations; LC fees spike |
| `Political_Discontinuity` | Government transition; LCY devalues, tariff uncertainty |
| `Commodity_FX_Correlation_Shock` | Correlated commodity + FX move: import cost double-hit |
| `Tariff_Circular` | New import duty circular; duty rate up 10 percentage points |
| `Severe_Stress` | Combined tail scenario across all shock types |

Custom scenarios are straightforward to add via `ScenarioShock`.

### `lc_priority.py` -- LC Priority Allocator

When USD capacity is constrained, ranks pending LC requests and produces an allocation plan (approve / defer / cancel) using a configurable scoring model:

| Factor | Default Weight | Description |
|---|---|---|
| Urgency | 40% | Operational impact of not opening this LC |
| Strategic value | 25% | Supplier relationship / long-term importance |
| Expiry pressure | 20% | Time until the LC window closes |
| Cost efficiency | 15% | Unit cost relative to peers |

## Quickstart

```bash
pip install -r requirements.txt
PYTHONPATH=. python examples/demo.py
```

The demo simulates a 6-month tightening cycle. Official FX rates drift while realized settlement rates diverge, the regime classifier escalates from NORMAL to STRESSED, scenario outputs show P&L exposure across shock types, and the LC allocator prioritizes a constrained USD pool.

## Project Structure

```
├── regime_risk/
│   ├── fx_curve.py           # Effective FX curve builder
│   ├── regime_classifier.py  # Non-market regime classifier
│   ├── scenario_engine.py    # Shock scenario engine
│   ├── lc_priority.py        # LC priority allocator
│   └── __init__.py
├── tests/                    # 78 tests, one file per module
├── examples/
│   └── demo.py               # End-to-end demo
├── KNOWN_ISSUES.md           # Defects found by the test suite
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
└── README.md
```

## Tests

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. python -m pytest
```

78 tests, one file per module. The scoring and P&L arithmetic in this framework
is simple enough to work out on paper, so the tests do that rather than freezing
whatever the code currently returns: expected values are derived from the
formulas in the class docstrings and from series whose correct answer is known by
construction.

The sharpest examples are the ones built on inputs with a known-zero answer. To
test the FX curve, realized settlement rates are set exactly equal to the
official rate on every date, so the true basis is identically zero and anything
the builder reports is its own artefact. That test is how issue #1 was found.

Writing the suite surfaced 16 defects, recorded in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md) with severity and a proposed fix. Two change
conclusions rather than just outputs:

- The curve builder manufactures an FX basis out of a pure trend in the official
  rate, and at a realistic trend it falsely flags most days as a stressed
  regime on a series containing no dislocation at all.
- The regime label on a date depends on volatility observed after that date, so
  it cannot be computed in real time. A consequence worth knowing separately:
  because the z-score is full-sample, the largest value attainable over `n`
  observations is `(n-1)/sqrt(n)`, so at the default threshold the volatility
  signal cannot fire at all for `n <= 5`.

Three more are cases of a documented field that is never read —
`Position.settlement_fx`, `entry_price_usd`, `quantity` — and one is a control
that silently does nothing: `min_priority_threshold` is only checked after
capacity runs out, so an LC scoring below it is approved whenever there is room.

Tests that assert a defect say so in the docstring and cite the issue number.
They pass by pinning current behaviour, so fixing a bug breaks its test, which is
the signal to delete both.

Nothing is fixed yet. The issues are recorded first so the published behaviour is
documented, and so a fix and its description land together.

## Requirements

Python 3.10+, pandas, numpy. Tests additionally need pytest and ruff
(`requirements-dev.txt`).

## Design Notes

**Why rule-based instead of ML for the regime classifier?**
In low-frequency, high-impact regime events there's rarely enough labeled training data for a supervised model. A well-specified rule-based classifier with calibrated weights is more interpretable, easier to audit, and more reliable under distribution shift -- which is precisely the condition you're trying to detect.

**Why volume-weighted FX rates?**
Larger LCs are more representative of the true market-clearing rate than smaller spot transactions. Volume-weighting prevents small, anomalous settlements from distorting the effective curve.

**Why a priority allocator instead of optimization?**
A linear program would maximize some objective (e.g. total notional allocated), but in practice the constraint isn't just USD -- it's also relationships, expiry windows, and operational urgency that don't reduce cleanly to a single objective. The scoring approach makes tradeoffs explicit and auditable.

The cost of that choice is that allocation is greedy first-fit, so a large
high-priority LC that does not fit is deferred while smaller ones behind it are
funded, and the leftover capacity is not reported. See
[KNOWN_ISSUES.md](KNOWN_ISSUES.md) #11.

## Scope

This is a clear implementation of a set of ideas about non-market risk, built to
be read and argued with. It is not a production risk system: there is no data
ingestion, no persistence, no position source of truth, and the calibrations in
`DEFAULT_SCENARIOS` are illustrative rather than estimated from a specific
market. The two high-severity issues above should be fixed before any of its
output is used to support a decision.

## License

MIT. See [LICENSE](LICENSE).
