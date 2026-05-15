# HydroSat Final Technical Proposal

## 1. Mission Fit

HydroSat addresses the final-round Clean Water task as an onboard-friendly inference problem: estimate turbidity and chlorophyll-a from multispectral scenes while respecting the practical constraints of space-computing workflows. Our goal is not only to predict two water-quality indicators, but to do so through a bounded runtime that can operate from mounted point requests and local image access without depending on a heavy ground-first vision stack.

Core inference framing:

```text
mounted point requests + multispectral scenes
-> local patch extraction
-> water-quality feature generation
-> target-specific inference
-> compact JSON products for downstream action
```

## 2. Final Implemented Runtime

The final frozen repository runtime is a compact tabular ensemble pipeline:

```text
point table + 12-band GeoTIFF
-> 24x24 point-centered patch extraction
-> spectral, ratio, spatial, and seasonal features
-> target-specific ensemble regressors
-> target-aware runtime calibration
-> Track 2 JSON output packaging
```

Implemented critical-path components:

- point-centered patch extraction from multispectral TIFF scenes
- handcrafted water-quality-oriented feature generation
- separate turbidity and chlorophyll-a model paths
- frozen ensemble artifacts under `artifacts/models/`
- self-describing runtime defaults under `artifacts/models/runtime_env_defaults.json`
- containerized `/input` -> `/output` execution via `run.sh`
- released-Area8 local evaluation using the official scoring formula
- a reproducible score-push harness for full rebuilds and late-stage tuning

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

## 3. Measured Final Evidence

After the organizers released the full Area8 imagery and truth JSON files, we evaluated the final runtime with the official formula:

- `NRMSE = RMSE / mean_truth`
- `parameter_score = (0.5 * max(0, R2) + 0.5 * max(0, 1 - NRMSE)) * 100`
- `algorithm_score = 0.5 * turbidity_score + 0.5 * chla_score`

Final released-Area8 results:

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

Stable tracked evidence:

- `docs/results/released_area8_scores.json`
- `docs/results/released_area8_scores.md`
- `docs/results/final_score_push_summary.json`
- `docs/results/final_score_push_summary.md`

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

## 5. Innovation And Differentiation

HydroSat’s core innovation is not a single heavy deep model. It is the way the inference path is structured for bounded execution and water-quality specificity.

Current differentiators:

- handcrafted spectral indices tailored to water-quality behavior
- separate turbidity and chlorophyll-a modeling paths
- unseen-area-first model selection followed by bounded target-specific tuning
- self-describing runtime defaults so the winning configuration can be reproduced without hidden shell settings
- optional CNN and regime-routing utilities kept outside the default submission-critical dependency chain
- deterministic containerized input/output behavior aligned to the competition platform

## 6. Application Value

The current baseline supports several practical scenarios:

- rapid triage of turbidity spikes after storms or sediment events
- chlorophyll-a-driven bloom surveillance
- prioritization of field verification in data-sparse regions
- compact product downlink when bandwidth is the main bottleneck
- satellite-ground workflows where only the most decision-relevant outputs should be transmitted first

## 7. Honest Final Positioning

Our final positioning is intentionally pragmatic:

- the repository now contains a real, reproducible, container-ready final runtime
- the released-Area8 local score improved materially through retraining and bounded target-specific runtime calibration
- chlorophyll-a is currently the stronger target, while turbidity remains the harder generalization problem under geographic shift

## 8. Future Roadmap

This repository should be viewed as a compact operational baseline today and a platform for stronger onboard water-intelligence work next.

Priority next steps:

1. improve turbidity robustness under unseen optical and geographic regimes
2. add uncertainty or quality flags
3. extend regime-aware routing and confidence-aware product logic
4. compress the deployment bundle further
5. add mission-facing selective downlink behavior

Final presentation assets:

- `docs/proposal/hydrosat_best_technical_proposal.pptx`
- `docs/proposal/presentation_script.md`
