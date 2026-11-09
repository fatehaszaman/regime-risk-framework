# Algorithm guide

This guide follows the current classifier, including its limitations. Complexity
is an operation-count model, not a latency benchmark or a validation claim.

## Effective FX basis construction

Implementation: [`EffectiveFXCurveBuilder.build`](../regime_risk/fx_curve.py).

```text
# Effective FX Basis / Trend-Invariant Smoothing
# Goal: estimate the realized settlement dislocation from the official FX rate.
# Input: N settlement records across D distinct dates; smoothing window W
# Output: D effective rates, bases, and stress flags
# Time: O(N log N + D) including the date sort
# Memory: O(N + D) peak additional memory

SORT settlements by date
GROUP settlements by date
FOR each date d:
    official[d] = MEAN(official rates reported for d)
    realized[d] = WEIGHTED_MEAN(realized rates, LC amounts)
    IF observation_count[d] < configured minimum:
        realized[d] = official[d]

    raw_basis[d] = realized[d] - official[d]

smoothed_basis = TRAILING_MEAN(raw_basis, W)
effective_rate = official + smoothed_basis
stress_flag = smoothed_basis > configured threshold
RETURN dates, official, effective_rate, smoothed_basis, stress_flag
```

The operation order is deliberate. The signal of interest is the settlement
dislocation, not the absolute FX level. If realized and official rates move
together, `raw_basis` is zero at every date and its trailing mean remains zero.
Smoothing the realized level first and subtracting the current official level
would instead compare a lagged value with an unlagged value, converting an
ordinary shared trend into a false basis.

Reconstructing `effective_rate` as `official + smoothed_basis` preserves the
reported identity `basis = effective_rate - official` at every date while
anchoring the smoothed dislocation to the current official rate. This lets the
curve reduce day-to-day settlement noise without creating stress signals from
level drift alone.

Current limitations remain explicit: `W` counts settlement-date rows rather
than calendar or business days, and the stress threshold is one-sided. These
are tracked separately as issues #2 and #3 in
[`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md).

## Threshold-based regime classification

Implementation: [`RegimeClassifier.classify`](../regime_risk/regime_classifier.py)
and `_near_policy_event` in the same file.

```text
# Regime Labels / Weighted Boolean Signals
# Goal: assign an interpretable regime label to each supplied date.
# Input: T dates and aligned FX, utilization, volatility arrays; E policy dates
# Output: T signal records with labels, scores, flags, and notes
# Time: O(T*(E+1))
# Memory: O(T+E) peak additional memory, including returned records

prior_vol = SHIFT(volatility, 1)
rolling_mean = TRAILING_MEAN(prior_vol, lookback=20, minimum=20)
rolling_std = TRAILING_STD(prior_vol, lookback=20, minimum=20) + epsilon
z = (volatility - rolling_mean) / rolling_std
FOR each date i:
    fx_flag = fx_basis[i] > configured threshold
    lc_flag = utilization[i] > configured threshold
    vol_flag = history is sufficient AND z[i] > configured threshold
    event_flag = any configured event lies within 3 calendar days
    score = SUM(fixed signal weights * flags)
    label = first descending threshold satisfied by score
    APPEND label, score, flags, and active-signal notes
RETURN records
```

Why: shifted rolling statistics scan T values once and exclude both the current
and future observations from each date's volatility baseline. Each date still
scans all E policy dates and allocates an E-element distance vector. Four signal
weights and four label thresholds are fixed constants. Output records and the
volatility z-score vector both grow with T.

Example: with no policy dates, that flag is false and classification is O(T).
Adding 1,000 policy dates does not become an indexed lookup; all are considered
for every classification date.

## Interpretation and edge cases

- **Point-in-time volatility:** each z-score uses at most 20 observations
  strictly before its date. Appending future data cannot revise an earlier
  signal, and the current observation cannot inflate its own baseline.
- **Cold start:** by default, the volatility signal remains false until 20 prior
  observations exist, making the 21st observation the first eligible signal.
  Twenty observations is an illustrative one-month heuristic, not an
  empirically calibrated optimum. The lookback and minimum are configurable
  observation counts, not calendar-day windows.
- **Calendar semantics:** proximity is symmetric and uses calendar days.
  Whether a future policy event was already announced is not checked here.
- **Input contract:** arrays should align with dates and contain finite values.
  Length mismatches can fail; NaN comparisons may silently suppress a signal.
- **Constant volatility:** epsilon prevents division by zero, but does not make
  the resulting threshold system empirically calibrated.

These notes document current behavior; they do not introduce an event-date
index or change the policy-event semantics.

## Greeks-based option P&L attribution

Implementation: [`attribute`](../regime_risk/options.py) and
[`ScenarioEngine._run_option`](../regime_risk/scenario_engine.py).

```text
# Greeks Attribution / Black-Scholes Revaluation
# Goal: explain an option's scenario P&L by delta, gamma, vega and theta.
# Input: one European option (S0, K, T, r, sigma0, kind, quantity) and a
#        scenario (spot shock, vol shift, horizon dt, FX shift)
# Output: FX, delta, gamma, vega, theta and residual P&L in LCY
# Time: O(1) per option, so O(P) for P option positions
# Memory: O(1) per option, O(P) for the result table

S1 = S0 * (1 + spot_shock);  sigma1 = sigma0 + vol_shift
V0 = quantity * BS(S0, K, T, r, sigma0)
V1 = quantity * BS(S1, K, T - dt, r, sigma1)
delta, gamma, vega, theta = BS_GREEKS(S0, K, T, r, sigma0)

dS = S1 - S0
delta_pnl = quantity * delta * dS
gamma_pnl = quantity * 0.5 * gamma * dS^2
vega_pnl  = quantity * vega * (sigma1 - sigma0)
theta_pnl = quantity * theta * dt
residual  = (V1 - V0) - (delta_pnl + gamma_pnl + vega_pnl + theta_pnl)

fx_pnl_lcy = V0 * fx_shift                  # V0 is a USD asset
each greek_pnl_lcy = greek_pnl * (F0 + fx_shift)
RETURN fx_pnl_lcy, greek P&L in LCY, residual in LCY
```

The LCY split rests on an identity:
`V1 * F1 - V0 * F0 = V0 * (F1 - F0) + (V1 - V0) * F1`. The first term is the
FX P&L and the second is split by the Greeks, so the components always sum to
the full revaluation. All Greeks are taken at the starting point, so the
residual collects the cross terms (vanna, the delta-theta interaction) and the
third-order spot term. For a spot-only move it is third order, and halving
`dS` cuts it by about eight; the tests check this.

Worked example (S=100, K=100, r=5%, sigma=20%, T=1, one call): value 10.4506,
delta 0.6368, gamma 0.01876, vega 37.52 per 1.00 of vol, theta -6.414 per year.
A move to S=104 and sigma=22% over 10 calendar days (dt = 10/365) gives delta
P&L 2.547, gamma P&L 0.150, vega P&L 0.750 and theta P&L -0.176. Those four sum
to 3.272 against a full revaluation of 3.222, leaving a residual of -0.050,
about 1.6% of the P&L. Elapsed time is calendar time (days/365) throughout,
matching `horizon_years` in the code; on a trading-day basis (10/252) theta
P&L would be -0.255 and the full change 3.136.
