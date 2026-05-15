# HydroSat Final Technical Proposal

## 1. Mission Fit

HydroSat is built for the final-round Clean Water task: estimate turbidity and chlorophyll-a from multispectral satellite imagery in a way that can still make sense in a space-computing context. Instead of treating the challenge as only a leaderboard regression problem, we frame it as a compact onboard inference problem:

```text
mounted point requests + mounted multispectral scenes
-> bounded local patch extraction
-> target-specific water-quality inference
-> compact JSON products for rapid decision support
```

That framing matters because the organizers explicitly asked teams to consider on-orbit compute constraints, deployment practicality, and efficient inference.

## 2. Final Implemented Baseline

The final frozen repository runtime is a CPU-first inference pipeline with the following structure:

```text
point table + 12-band GeoTIFF
-> 32x32 point-centered patch extraction
-> spectral, ratio, spatial, and seasonal feature generation
-> target-specific ensemble regressors
-> released-stat calibration
-> Track 2 JSON output packaging
```

What is implemented in the repository today:

- point-centered patch extraction from 12-band TIFF imagery
- handcrafted water-quality-oriented feature generation
- separate turbidity and chlorophyll-a model paths
- final ensemble `.joblib` runtime bundle under `artifacts/models/`
- containerized execution for `/input`, `/output`, and `/workspace`
- released-Area8 local evaluator using the official final-round formula
- score-push sweep tooling for reproducible retraining and comparison

The final frozen runtime path uses:

- patch size: `32`
- selected retrain experiment: `area_filter10_score`
- area-grouped selection for the deployment candidate
- turbidity runtime mode: `blend`
- turbidity blend weight: `0.81`
- released-stat calibration enabled for both targets

## 3. Measured Evidence

After the organizers released the full Area8 image bundle and truth JSONs, we ran a full offline evaluation using the official final-round scoring formula:

- `NRMSE = RMSE / mean_truth`
- `parameter_score = (0.5 * max(0, R2) + 0.5 * max(0, 1 - NRMSE)) * 100`
- `algorithm_score = 0.5 * turbidity_score + 0.5 * chla_score`

Frozen submission-runtime results from `python -m hydrosat.evaluate_released_area8`:

- Turbidity: `RMSE 2.1440`, `R2 0.1155`, `NRMSE 0.9940`, `score 6.0765`
- Chl-a: `RMSE 1.1400`, `R2 0.0906`, `NRMSE 0.7055`, `score 19.2541`
- Final algorithm score: `12.6653`

This is a substantial improvement over the earlier frozen local baseline of `6.0453`, but it is not the strongest final-round technical direction once the released Area8 calibration set is available.

Released Area8 evaluation workload:

- `372` turbidity points
- `103` chl-a points
- `475` total points

Frozen runtime footprint:

- frozen runtime model bundle: `29.48 MB`

### Post-release site-calibrated research result

The strongest defensible local result is not truth replay. It is a site-calibrated model evaluated on held-out dates, so every validation date is unseen while the model may use calibration history from other dates at the same Area8 stations.

Local validation protocol:

- group split by acquisition date
- three date-held-out folds
- no held-out labels used during fitting
- same official Track 2 score formula
- released Area8 imagery used for feature extraction

Best honest date-held-out result from `scripts/site_calibration_cv.py`:

- Turbidity: `score 54.0113`
- Chl-a: `score 53.8813`
- Algorithm score: `53.9463`

Best system shape:

```text
spectral-spatial ExtraTrees model
+ same-site temporal calibration prior
+ per-target blending strategy
```

For turbidity, the best local blend uses a learned spectral model plus per-site linear interpolation across surrounding observations. For chlorophyll-a, the best local blend uses a learned spectral model plus nearest-date same-site seasonal priors. This is much stronger than the frozen generic runtime and still avoids the invalid `100`-score shortcut of copying released truths into outputs.

## 4. Why This Is Feasible For On-Orbit Computing

The final critical path is intentionally CPU-first:

- it does not require CNN inference to succeed
- it ships compact pretrained ensemble artifacts instead of a large end-to-end vision stack
- it processes point requests and local patches, not full-scene tensors on the main path
- it produces compact JSON products instead of large derived raster payloads

From a mission perspective, this supports a reasonable onboard interpretation:

- ingest requested coordinates
- read only the needed local pixels
- infer water-quality indicators
- downlink compact products for faster triage

This is not yet a flight-qualified system, but it is a much more portable and auditable baseline than a GPU-only submission path.

## 5. Innovation And Differentiation

The strongest innovation in HydroSat is not a single large deep model. It is the way the inference path is structured for bounded execution and water-quality specificity.

Current differentiators:

- handcrafted spectral indices tailored to water-quality behavior rather than generic image embeddings alone
- separate turbidity and chlorophyll-a modeling paths
- explicit score-push retraining workflow focused on unseen-area behavior
- optional CNN path kept outside the default runtime dependency chain
- deterministic containerized input and output contract aligned to the competition platform
- optional site-calibration layer that adapts generic spectral inference to a monitored basin without leaking held-out dates
- temporal continuity modeling for repeat monitoring stations, which is directly relevant to operational water-quality surveillance

This makes HydroSat different from a purely ground-first workflow where whole scenes are downlinked first and interpreted later.

## 6. Application Value

HydroSat can support several practical water-quality scenarios:

- rapid triage of turbidity spikes after storms or sediment events
- chlorophyll-a driven bloom surveillance
- faster prioritization of field verification in data-sparse regions
- compact product downlink when bandwidth is the bottleneck
- satellite-ground workflows where only the most decision-relevant outputs should be transmitted first

The social value is strongest in settings where in situ sampling is sparse and water-quality changes need early warning rather than perfect retrospective mapping.

## 7. Honest Final Positioning

Our final positioning is intentionally honest:

- the repository contains a real, reproducible, container-ready frozen runtime
- the frozen submission score is only `12.6653` on full released Area8 and should not be oversold
- the stronger final-round concept is the post-release site-calibrated system, which reaches `53.9463` under date-held-out validation without using held-out truths
- the invalid truth-replay path can score `100`, but it is leakage and is not part of the proposal
- turbidity remains the limiting target under geographic shift, so further gains should focus there first

## 8. Future Roadmap

The roadmap now starts from a stronger frozen baseline rather than from a partial prototype.

Priority next steps:

1. productionize the site-calibration layer with uncertainty-aware fallbacks
2. improve turbidity robustness on unseen regions and optical regimes
3. add explicit uncertainty or quality flags
4. introduce regime-aware routing or confidence-gated inference
5. further compress the deployment bundle
6. add mission-facing selective downlink logic

So the current repository should be understood as a compact operational baseline today and a platform for a more capable onboard water-intelligence system next.
