# HydroSat Final Technical Proposal

## 1. Mission Fit

HydroSat is designed for the Clean Water final-round task: estimate turbidity and chlorophyll-a from multi-band satellite imagery while staying compatible with an onboard-computing context. The implemented baseline favors a small, auditable, CPU-first inference path rather than a GPU-only design, because the competition brief emphasizes constrained space-computing environments.

## 2. Implemented Baseline System

The checked-in system is a deterministic inference pipeline:

```text
point table + 12-band GeoTIFF
-> point-centered patch extraction
-> spectral and spatial feature generation
-> target-specific ensemble regressors
-> calibration and clipping
-> JSON output packaging
```

What is implemented in the repository today:

- point-centered patch extraction from Sentinel-style 12-band TIFFs
- 32x32 patch-based handcrafted spectral and texture features
- separate turbidity and chl-a inference paths
- ensemble `.joblib` models as the default runtime path
- optional CNN artifacts retained outside the critical path
- Dockerized execution for `/input`, `/output`, and `/workspace`

The current feature builder emits about `1073` features per point, with `1070` numeric features.

## 3. Measured Local Evidence

After the organizers released the full Area8 image bundle and truth JSONs, the repository was evaluated offline using the official final-round scoring formula from the task PDF:

- `NRMSE = RMSE / mean_truth`
- `parameter_score = (0.5 * max(0, R2) + 0.5 * max(0, 1 - NRMSE)) * 100`
- `algorithm_score = 0.5 * turbidity_score + 0.5 * chla_score`

Measured results from `python -m hydrosat.evaluate_released_area8`:

- Turbidity: `RMSE 2.4604`, `R2 -0.1649`, `score 0.0000`
- Chl-a: `RMSE 1.2252`, `R2 -0.0503`, `score 12.0906`
- Final algorithm score: `6.0453`

The released Area8 evaluation processed:

- `372` turbidity points
- `103` chl-a points
- `475` total points

Runtime on this local RTX 5070 laptop workflow was about `20.43` seconds end to end. The default ensemble inference bundle is about `29.57 MB`, while the full checked-in model folder including optional CNN artifacts is about `73.01 MB`.

## 4. On-Orbit Feasibility

The implemented baseline is more plausible for on-orbit deployment than a CNN-only stack because:

- the critical path does not require GPU acceleration
- serialized ensemble weights are compact compared with large end-to-end vision models
- inference streams point rows and local patches instead of retaining whole-scene tensors in memory
- the execution path is deterministic, bounded, and easy to audit

In a mission setting, this style of model can support:

- onboard request-table processing
- compact downlink of water-quality products rather than full-scene imagery
- fallback-safe execution when accelerators are unavailable

## 5. Innovation and Differentiation

The current codebase is strongest as a flight-oriented spectral inference baseline:

- it uses water-quality-specific handcrafted indices instead of only generic image embeddings
- it separates turbidity and chl-a inference so each target can use different feature importance patterns
- it supports optional neural augmentation without making CNNs the only runtime path
- it aligns to the competition's containerized onboard-computing framing

The main future-looking innovation path, not yet fully implemented in the checked-in baseline, is to add:

- explicit quality gating
- optical-regime routing
- uncertainty estimation
- selective downlink logic for only the highest-value outputs or patches

Those items are presented as roadmap elements rather than current code claims.

## 6. Value and Application Scenarios

A system of this type can support:

- rapid detection of deteriorating inland water quality
- tracking of turbidity spikes after storms or sediment events
- early warning for bloom-like chl-a behavior
- selective downlink in bandwidth-constrained satellite operations
- downstream decision support for water utilities, environmental agencies, and emergency response teams

The social value is strongest where field sampling is sparse and water-quality changes must be triaged quickly.

## 7. Future Roadmap

The next technical steps are:

1. improve turbidity generalization on unseen regions
2. calibrate both targets using stronger distribution-shift handling
3. add explicit uncertainty outputs
4. compress the deployed model bundle further
5. add mission-style downlink prioritization logic

Our final positioning should be honest and strong: the repository already contains a runnable, containerized, CPU-first water-quality inference baseline, and the next stage is to turn that baseline into a more robust onboard decision system with uncertainty-aware selective downlink.
