# Submission Manifest

## Runnable Competition Package

The runnable package currently present in this repository is:

```text
hydrosat_submission_public_image_direct_500.zip
```

This is the strongest packaged submission artifact that was actually exported and preserved locally. Its previous local validation estimate was:

```text
Algorithm score: 57.1568
```

## Final Proposed Technical Package

The final technical solution is the hybrid regime-aware model described in `FINAL_TECHNICAL_PROPOSAL.md`.

Its best local validation result was:

```text
Algorithm score: 57.6381
```

Best blend:

```text
turbidity_alpha = 0.20
chla_alpha      = 0.03
```

This corresponds to:

```text
turbidity_final = 0.80 * global_turbidity + 0.20 * regime_turbidity
chla_final      = 0.97 * global_chla      + 0.03 * regime_chla
```

## Packaging Status

A new runnable hybrid submission zip has not yet been exported into this repository. The proposed package name should be:

```text
hydrosat_submission_hybrid_regime_5764.zip
```

To create that final zip, the Kaggle hybrid artifacts must be wired into `hydrosat.infer`, replacing the existing packaged inference behavior with the global-plus-regime blend.

Until that packaging step is completed, the repository contains:

- `hydrosat_submission_public_image_direct_500.zip` as the preserved runnable package.
- `FINAL_TECHNICAL_PROPOSAL.md` as the final proposed solution.
- `hydrosat_best_technical_proposal.pptx` as the proposal deck.

