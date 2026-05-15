# Site-Adaptive Research Summary

This note documents the post-release research layer separately from the frozen competition runtime.

## Question

If HydroSat is monitoring a basin that already has local historical measurements, can same-site temporal adaptation materially improve monitored-site products?

## Retrospective Date-Held-Out Result

- Turbidity score: `54.0113`
- Chl-a score: `53.8813`
- Algorithm score: `53.9463`

## Best Research Shape

```text
spectral-spatial ExtraTrees model
+ same-site temporal calibration prior
+ per-target blend strategy
```

- Turbidity uses spectral inference plus same-site interpolation across surrounding observations.
- Chl-a uses spectral inference plus nearby-date same-site seasonal priors.

## Interpretation

This is **not** the frozen leaderboard-equivalent score. It is a retrospective site-adaptive monitoring result that demonstrates the value of local calibration history once a basin has been observed repeatedly.

The frozen runtime remains the correct public score for the actual submission bundle:

- Frozen runtime algorithm score: `14.5762`

## Reproduction

- Script: `scripts/site_calibration_cv.py`
- Inputs: released Area8 imagery and truth JSONs
- Split design: date-held-out grouped validation
