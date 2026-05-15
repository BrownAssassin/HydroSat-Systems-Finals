# HydroSat Final Technical Proposal

## 1. Mission Fit

HydroSat is built for the final-round Clean Water task: estimate turbidity and chlorophyll-a from multispectral satellite imagery in a way that still makes sense in an on-orbit computing context. Instead of treating the challenge only as a leaderboard regression problem, we frame it as a bounded onboard inference problem:

```text
mounted point requests + mounted multispectral scenes
-> bounded local patch extraction
-> target-specific water-quality inference
-> compact JSON products for rapid decision support
```

That framing matters because the organizers explicitly asked teams to address computational limits, deployment practicality, and efficient inference in a space environment.

## 2. Final Implemented Runtime

The final frozen repository runtime is a compact inference pipeline with the following structure:

```text
point table + 12-band GeoTIFF
-> 24x24 point-centered patch extraction
-> spectral, ratio, spatial, and seasonal feature generation
-> target-specific ensemble regressors
-> target-aware runtime calibration
-> Track 2 JSON output packaging
```

What is implemented in the repository today:

- point-centered patch extraction from 12-band TIFF imagery
- handcrafted water-quality-oriented feature generation
- separate turbidity and chlorophyll-a model paths
- final ensemble runtime bundle under `artifacts/models/`
- self-describing runtime defaults in `artifacts/models/runtime_env_defaults.json`
- containerized execution for `/input`, `/output`, and `/workspace`
- released-Area8 local evaluator using the official final-round formula
- a reproducible score-push harness for tabular retraining, pairing, and bounded runtime tuning

The final frozen runtime path uses:

- winning pair: `patch24_filter15_top3_feat800__patch24_filter15_top3_feat800`
- stage A best structure: `patch24_filter15`
- stage B best experiment: `patch24_filter15_top3_feat800`
- patch size: `24`
- turbidity runtime mode: `model`
- turbidity calibration: `lognormal_rank`
- turbidity lognormal sigma: `0.52`
- turbidity prior shrink: `0.05`
- chlorophyll-a runtime mode: `model`
- released-stat calibration enabled for both targets

## 3. Measured Final Evidence

After the organizers released the full Area8 image bundle and truth JSONs, we ran a full offline evaluation using the official final-round scoring formula:

- `NRMSE = RMSE / mean_truth`
- `parameter_score = (0.5 * max(0, R2) + 0.5 * max(0, 1 - NRMSE)) * 100`
- `algorithm_score = 0.5 * turbidity_score + 0.5 * chla_score`

Final frozen results from `python -m hydrosat.evaluate_released_area8` after the last late-stage tabular push:

- Turbidity: `RMSE 2.0728`, `R2 0.1733`, `NRMSE 0.9609`, `score 10.6170`
- Chl-a: `RMSE 1.1465`, `R2 0.0802`, `NRMSE 0.7095`, `score 18.5354`
- Final algorithm score: `14.5762`

Improvement over the prior frozen baseline:

- previous frozen runtime score: `14.4445`
- final released-Area8 algorithm score: `14.5762`
- late-stage gain: `+0.1318` points (`+0.91%`)
- total gain over the earlier `12.6653` baseline: `+1.9109` points (`+15.09%`)

Released Area8 evaluation workload:

- `372` turbidity points
- `103` chl-a points
- `475` total points

Final runtime footprint:

- frozen runtime model bundle: `4.84 MB`

Measured local runtime:

- full released-Area8 evaluation: `25.23 s`

## 4. Why This Is Feasible For On-Orbit Computing

The final critical path is intentionally CPU-first:

- it does not require CNN inference to succeed
- it ships compact pretrained ensemble artifacts instead of a large end-to-end vision stack
- it processes point requests and local patches, not full-scene tensors on the main path
- it produces compact JSON products instead of large derived raster payloads

From a mission perspective, this supports a practical onboard interpretation:

- ingest requested coordinates
- read only the needed local pixels
- infer water-quality indicators
- downlink compact products for faster triage

This is not yet a flight-qualified system, but it is far more portable and auditable than a GPU-only submission path.

## 5. Innovation And Differentiation

The strongest innovation in HydroSat is not a single large deep model. It is the way the inference path is structured for bounded execution and water-quality specificity.

Current differentiators:

- handcrafted spectral indices tailored to water-quality behavior rather than generic image embeddings alone
- separate turbidity and chlorophyll-a modeling paths
- a multi-stage search harness that first narrows by unseen-area CV and then applies bounded released-Area8 tuning
- a self-describing runtime bundle so the winning configuration can be reproduced without hidden shell settings
- optional CNN and regime-ensemble paths kept outside the default submission dependency chain
- deterministic containerized input and output contract aligned to the competition platform

This makes HydroSat meaningfully different from a purely ground-first workflow where full scenes are downlinked first and interpreted later.

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

- the repository now contains a real, reproducible, container-ready final runtime
- the released-Area8 local score improved materially through retraining and bounded target-specific runtime calibration
- chlorophyll-a is now the stronger target, but turbidity remains the harder generalization problem under geographic shift

## 8. Future Roadmap

The roadmap now starts from a stronger frozen baseline rather than from a partial prototype.

Priority next steps:

1. improve turbidity robustness under unseen optical and geographic regimes
2. add explicit uncertainty or quality flags
3. extend regime-aware routing and confidence-aware product logic
4. further compress the deployment bundle
5. add mission-facing selective downlink behavior

The current repository should therefore be understood as a compact operational baseline today and a platform for a more capable onboard water-intelligence system next.
