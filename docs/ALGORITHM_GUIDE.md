# Algorithm guide

This guide follows the current classifier, including its limitations. Complexity
is an operation-count model, not a latency benchmark or a validation claim.

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
