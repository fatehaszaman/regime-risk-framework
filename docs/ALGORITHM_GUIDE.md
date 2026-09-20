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

mean_vol = MEAN(all supplied volatility values)
std_vol = STD(all supplied volatility values) + epsilon
z = (volatility - mean_vol) / std_vol
FOR each date i:
    fx_flag = fx_basis[i] > configured threshold
    lc_flag = utilization[i] > configured threshold
    vol_flag = z[i] > configured threshold
    event_flag = any configured event lies within 3 calendar days
    score = SUM(fixed signal weights * flags)
    label = first descending threshold satisfied by score
    APPEND label, score, flags, and active-signal notes
RETURN records
```

Why: normalization scans T values; each date scans all E policy dates and
allocates an E-element distance vector. Four signal weights and four label
thresholds are fixed constants. Output records and the volatility z-score vector
both grow with T.

Example: with no policy dates, that flag is false and classification is O(T).
Adding 1,000 policy dates does not become an indexed lookup; all are considered
for every classification date.

## Interpretation and edge cases

- **Historical use:** normalization uses the entire supplied volatility array,
  not only history available at each date. Passing a full backtest period can
  therefore introduce look-ahead. This is not a point-in-time rolling classifier.
- **Calendar semantics:** proximity is symmetric and uses calendar days.
  Whether a future policy event was already announced is not checked here.
- **Input contract:** arrays should align with dates and contain finite values.
  Length mismatches can fail; NaN comparisons may silently suppress a signal.
- **Constant volatility:** epsilon prevents division by zero, but does not make
  the resulting threshold system empirically calibrated.

These notes document current behavior; they do not introduce a rolling
normalization or event-date index.
