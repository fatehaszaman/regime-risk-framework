# Regime Risk Framework

[Algorithm guide: pseudocode, time complexity, and memory](docs/ALGORITHM_GUIDE.md).

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

The builder calculates the realized-versus-official basis first and smooths
that dislocation rather than the FX level. A shared trend in official and
realized rates therefore remains a zero basis instead of producing a false
stress signal. The separate row-versus-calendar-day window limitation remains
documented in [KNOWN_ISSUES.md](KNOWN_ISSUES.md) #2.

### `regime_classifier.py` -- Regime Classifier

Rule-based classifier that assigns a regime label (NORMAL / ELEVATED / STRESSED / CRISIS) at each timestep based on four signals:

| Signal | Weight | Description |
|---|---|---|
| FX basis | 35% | Spread between effective and official FX rate |
| LC utilization | 30% | Fraction of USD allocation capacity in use |
| Policy event | 20% | Proximity to known policy discontinuity dates |
| Volatility | 15% | Point-in-time z-score against prior realized vol |

The volatility baseline uses 20 observations strictly before each date, so the
first eligible signal is the 21st observation. This is an illustrative
one-month observation window, not an empirically calibrated optimum; both the
lookback and minimum history are configurable. Future data and the current
observation cannot revise or dilute the signal being scored.

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
├── tests/                    # Arithmetic and behavioral regression tests
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

Tests are organized by module. The scoring and P&L arithmetic in this framework
is simple enough to work out on paper, so the tests do that rather than freezing
whatever the code currently returns: expected values are derived from the
formulas in the class docstrings and from series whose correct answer is known by
construction.

The sharpest examples are built on inputs with a known-zero answer. For the FX
curve, realized settlement rates are set exactly equal to the official rate on
every date while both trend. The regression tests require a zero basis and no
stress flags throughout.

The allocation threshold now applies regardless of available capacity, unused
capacity is reported, and invalid weights fail even under `python -O`.
Scenario calculations use each position's settlement FX, floor fee and tariff
levels at zero, and reconcile notional to quantity times entry price.
Hand-calculated regression tests cover these corrections.

The scenario engine remains an additive invoice-notional sensitivity model,
not a full landed-cost repricer. Existing fee/tariff amounts are not revalued
under FX and commodity changes. Its precise formula and the remaining issues
are documented in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

Tests that assert an unresolved defect say so in the docstring and cite the
issue number. When a bug is fixed, the pinning test is replaced with a
regression test for the corrected result.

Remaining issues stay explicit until their fixes and regression tests land.

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
market. The remaining high-severity issues should be fixed before any of its
output is used to support a decision.

## License

MIT. See [LICENSE](LICENSE).
