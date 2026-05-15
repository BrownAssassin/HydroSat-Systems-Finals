# HydroSat Final Technical Proposal

## 1. Mission Fit

HydroSat addresses the final-round Clean Water task as an onboard-friendly inference problem: estimate turbidity and chlorophyll-a from multispectral scenes while respecting the practical constraints of space-computing workflows. Our goal is not only to predict two water-quality indicators, but to build a system that can begin with a compact frozen runtime and become more useful over time as a monitored basin accumulates local calibration history.

Core mission framing:

```text
mounted point requests + multispectral scenes
-> local patch extraction
-> water-quality feature generation
-> target-specific inference
-> optional site-adaptive calibration
-> compact products for downstream action
```

## 2. Two-Layer System Design

### Layer A: Frozen onboard runtime

The submission-critical path is a compact tabular ensemble pipeline:

```text
point table + 12-band GeoTIFF
-> 24x24 point-centered patch extraction
-> spectral, ratio, spatial, and seasonal features
-> target-specific ensemble regressors
-> target-aware runtime calibration
-> Track 2 JSON output packaging
```

Implemented frozen-runtime components:

- point-centered patch extraction from multispectral TIFF scenes
- handcrafted water-quality-oriented feature generation
- separate turbidity and chlorophyll-a model paths
- frozen ensemble artifacts under `artifacts/models/`
- self-describing runtime defaults under `artifacts/models/runtime_env_defaults.json`
- containerized `/input` -> `/output` execution via `run.sh`
- released-Area8 local evaluation using the official scoring formula

Final frozen winner:

- structure family: `patch24_filter15`
- final paired bundle: `patch24_filter15_top3_feat800__patch24_filter15_top3_feat800`
- patch size: `24`
- turbidity mode: `model`
- turbidity calibration: `lognormal_rank`
- turbidity sigma: `0.52`
- turbidity prior shrink: `0.05`
- chl-a mode: `model`
- released-stat calibration enabled
- CNN disabled on the critical runtime path

### Layer B: Site-adaptive monitoring

The stronger proposed mission architecture adds a second layer when a basin has historical local measurements available:

```text
frozen spectral inference
+ historical same-site calibration
+ temporal continuity model
-> refined monitored-site product
```

This layer treats repeat stations as time series rather than isolated points. It can use prior local measurements to adapt a generic spectral model to a monitored basin, which is operationally useful because environmental monitoring systems often become more valuable after deployment as local matchup history accumulates.

## 3. Honest Performance Story

### Frozen competition-runtime evidence

After the organizers released the full Area8 imagery and truth JSON files, we evaluated the final runtime with the official formula:

- `NRMSE = RMSE / mean_truth`
- `parameter_score = (0.5 * max(0, R2) + 0.5 * max(0, 1 - NRMSE)) * 100`
- `algorithm_score = 0.5 * turbidity_score + 0.5 * chla_score`

Frozen released-Area8 results:

- Turbidity: `RMSE 2.0728`, `R2 0.1733`, `NRMSE 0.9609`, `score 10.6170`
- Chl-a: `RMSE 1.1465`, `R2 0.0802`, `NRMSE 0.7095`, `score 18.5354`
- Final algorithm score: `14.5762`

Improvement history:

- earlier frozen baseline: `12.6653`
- prior late-stage frozen runtime: `14.4445`
- final frozen runtime: `14.5762`
- total gain over the `12.6653` baseline: `+1.9109` points (`+15.09%`)

Evaluation workload and footprint:

- `372` turbidity points
- `103` chl-a points
- `475` total evaluation points
- runtime bundle size: `4.84 MB`
- measured local released-Area8 runtime: `25.23 s`

Tracked frozen-runtime evidence:

- `docs/results/released_area8_scores.json`
- `docs/results/released_area8_scores.md`
- `docs/results/final_score_push_summary.json`
- `docs/results/final_score_push_summary.md`

### Site-adaptive research evidence

The stronger post-release research layer is documented separately because it answers a different operational question from the leaderboard. It asks:

> Once HydroSat is monitoring a basin that already has local calibration history, how much can local temporal adaptation improve monitored-site products?

In a post-release retrospective date-held-out study using the released Area8 truth set:

- Turbidity score: `54.0113`
- Chl-a score: `53.8813`
- Algorithm score: `53.9463`

Best research configuration:

```text
spectral-spatial ExtraTrees model
+ same-site temporal calibration prior
+ per-target blend strategy
```

For turbidity, the strongest retrospective blend combines spectral inference with same-site interpolation across surrounding observations. For chlorophyll-a, the strongest blend combines spectral inference with nearby-date seasonal priors. This is **not** the frozen leaderboard-equivalent result and should not be presented as such. It is evidence that HydroSat becomes far more useful as a site-adaptive monitoring system once local history exists.

Tracked site-adaptive evidence:

- `scripts/site_calibration_cv.py`
- `docs/results/site_adaptive_research.md`
- `docs/results/site_adaptive_research.json`

## 4. Why This Is Feasible For On-Orbit Computing

The final critical path is intentionally CPU-first and compact:

- it does not require CNN inference to succeed
- it ships small pretrained ensemble artifacts rather than a large end-to-end deep vision stack
- it processes point requests and bounded local patches instead of full-scene tensors
- it produces compact JSON outputs that are practical for selective downlink workflows

This supports a realistic onboard interpretation:

- ingest requested coordinates
- read only the needed local pixels
- infer the most decision-relevant water-quality indicators
- transmit compact products for faster triage on the ground

The site-adaptive layer is complementary rather than contradictory. The spacecraft can still run the lightweight frozen estimator, while ground-side calibration updates or future onboard parameter updates can refine monitored-site products as history accumulates.

## 5. Innovation And Differentiation

HydroSat’s innovation is the combination of a deployable baseline and an adaptive monitoring roadmap:

- handcrafted spectral indices tailored to water-quality behavior
- separate turbidity and chlorophyll-a modeling paths
- unseen-area-first model selection followed by bounded target-specific tuning
- self-describing runtime defaults so the winning configuration can be reproduced without hidden shell settings
- optional site-adaptive calibration for basins with local monitoring history
- temporal continuity modeling for repeat monitoring stations
- optional CNN and regime-routing utilities kept outside the default submission-critical dependency chain
- deterministic containerized input/output behavior aligned to the competition platform

## 6. Application Value

HydroSat supports two useful operating modes:

1. **Cold-start monitoring**
   - deploy the frozen runtime immediately over a new region
   - downlink compact preliminary water-quality products
   - flag lower-confidence cases for review or follow-up

2. **Mature monitored-basin operations**
   - combine current imagery with historical local calibration data
   - stabilize repeated station products over time
   - improve triage for turbidity spikes, bloom surveillance, and field-verification planning

## 7. Honest Final Positioning

Our final positioning is intentionally explicit:

- the repository contains a real, reproducible, container-ready frozen runtime
- the frozen leaderboard-comparable released-Area8 result is `14.5762`
- the stronger proposed architecture is a site-adaptive monitoring system, not a claim that the frozen leaderboard model scored `53+`
- the post-release site-adaptive study reached `53.9463` under retrospective date-held-out evaluation
- that research result demonstrates the value of local calibration history, while future causal-only forecasting remains a separate validation target
- direct truth replay is leakage and is not part of the proposal

## 8. Future Roadmap

Priority next steps:

1. productionize the site-adaptive layer with uncertainty-aware fallbacks
2. test causal forward-only adaptation for near-real-time operations
3. improve turbidity robustness under unseen optical and geographic regimes
4. extend regime-aware routing and confidence-aware product logic
5. compress the deployment bundle further
6. add mission-facing selective downlink behavior

Final presentation assets:

- `docs/proposal/hydrosat_best_technical_proposal.pptx`
- `docs/proposal/presentation_script.md`
