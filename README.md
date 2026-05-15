# HydroSat Systems - Track 2 Final Round

HydroSat Systems is our final-round repository for `Track 2: Space Intelligence Promoting Water Quality` in the ITU AI and Space Computing Challenge. The final frozen submission path is a compact inference pipeline that predicts:

- turbidity
- chlorophyll-a

from 12-band GeoTIFF imagery using point-centered patch extraction, handcrafted spectral-spatial features, and target-specific ensemble regressors.

## Final Frozen Runtime

The current default runtime is the best validated late-stage score-push candidate:

- winning pair: `patch24_filter15_top3_feat800__patch24_filter15_top3_feat800`
- stage A best structure: `patch24_filter15`
- stage B best experiment: `patch24_filter15_top3_feat800`
- feature patch size: `24x24`
- turbidity runtime mode: `model`
- turbidity calibration: `lognormal_rank`
- turbidity lognormal sigma: `0.52`
- turbidity prior shrink: `0.05`
- chlorophyll-a runtime mode: `model`
- released-stat calibration: enabled
- CNNs: disabled on the critical path

Runtime model files:

- `artifacts/models/turbidity_ensemble.joblib`
- `artifacts/models/chla_ensemble.joblib`
- `artifacts/models/runtime_env_defaults.json`

Frozen runtime artifact footprint:

- `4.84 MB`

## Final Released Area8 Offline Evaluation

Using the official released Area8 truth JSONs and the final-round scoring formula:

- Turbidity: `RMSE = 2.0728`, `R2 = 0.1733`, `NRMSE = 0.9609`, `score = 10.6170`
- Chl-a: `RMSE = 1.1465`, `R2 = 0.0802`, `NRMSE = 0.7095`, `score = 18.5354`
- Algorithm score: `14.5762`

Improvement over the prior frozen baseline:

- previous frozen runtime score: `14.4445`
- late-stage gain: `+0.1318` points (`+0.91%`)
- total gain over the earlier `12.6653` baseline: `+1.9109` points (`+15.09%`)

Those final numbers are written to:

- `artifacts/reports/released_area8/released_area8_scores.json`
- `artifacts/reports/released_area8/released_area8_scores.md`

The final late-stage search summary is written to:

- `artifacts/reports/final_two_hour_score_push_summary.json`
- `artifacts/reports/final_two_hour_score_push_summary.md`

## Repo Layout

- `src/hydrosat/` contains raster IO, feature extraction, training CLIs, inference CLI, released-Area8 scoring, and the final score-push harness.
- `artifacts/models/` contains the frozen runtime models used by inference.
- `track2_download_link_1/` to `track2_download_link_5/` contain the raw organizer downloads and stay at the repo root for compatibility.
- `Hydro Sat Systems_Arv Bali_baliarv21@gmail.com/` contains the regenerated deck and presentation script.
- `scripts/build_proposal_deck.py` regenerates the PPT from the current released-Area8 report.

Generated local outputs are ignored under:

- `artifacts/eval_input/`
- `artifacts/output/`
- `artifacts/reports/`
- `artifacts/features/`
- `artifacts/experiments/`

## Local Setup On This Windows GPU Machine

Use Python `3.10`.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Quick verification:

```powershell
python -c "import sklearn, torch; print('sklearn', sklearn.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected local state:

- `scikit-learn 1.7.2`
- CUDA available: `True`
- device: `NVIDIA GeForce RTX 5070 Laptop GPU`

## Run Sample Input Inference

The runtime bundle now carries its own defaults in `artifacts/models/runtime_env_defaults.json`, so the standard CLI reproduces the winning configuration automatically.

```powershell
$env:PYTHONPATH = "src"
python -m hydrosat.infer `
  --input-root "track2_download_link_1\Guide to the Second Round_track2\test_input_sample" `
  --model-dir "artifacts\models" `
  --output-dir "artifacts\output\sample_input_final" `
  --progress-every 1000
```

This writes all four naming variants:

- `turbidity_result.json`
- `chla_result.json`
- `result_turbidity.json`
- `result_chla.json`

## Re-Run Final Released Area8 Evaluation

The released-Area8 evaluation is the source of truth for the final local score:

```powershell
$env:PYTHONPATH = "src"
python -m hydrosat.evaluate_released_area8 `
  --released-root "track2_download_link_1" `
  --model-dir "artifacts\models" `
  --work-dir "artifacts\eval_input\released_area8" `
  --output-dir "artifacts\output\released_area8" `
  --report-dir "artifacts\reports\released_area8" `
  --patch-size 24 `
  --progress-every 1000
```

Current measured workload:

- `372` turbidity points
- `103` chl-a points
- `475` total points
- measured local runtime: `25.23 s`

## Reproduce The Score Push Harness

Build features for all searched patch sizes:

```powershell
$env:PYTHONPATH = "src"
python -m hydrosat.score_push `
  --data-root . `
  --released-root track2_download_link_1 `
  --features-dir artifacts/features `
  --experiments-dir artifacts/experiments `
  --reports-dir artifacts/reports `
  --runtime-model-dir artifacts/models `
  --progress-every 500 `
  --random-state 42 `
  --selection-metric score
```

What the final score-push harness now does:

- builds features for patch sizes `24`, `32`, and `40`
- runs Stage A structural search over `full`, `filter05`, `filter10`, `filter15`, `filter10_clip97`, and `filter10_clip99`
- runs Stage B fine search over the best promoted structures
- tunes the current best tabular winner with a bounded released-Area8 last-mile search
- evaluates the saved regime turbidity candidate against the best current chl-a bundle
- runs a narrow turbidity-only retrain fallback if tuning alone does not beat the live floor
- freezes the best winning runtime back into `artifacts/models/`

The final winner before freezing in the last push was:

- `patch24_filter15_top3_feat800__patch24_filter15_top3_feat800`

## Submission Notes

- `Dockerfile` uses the competition PyTorch CUDA base image.
- `.gitlab-ci.yml` is aligned to `./run.sh`.
- `run.sh` expects:
  - `INPUT_DIR=/input`
  - `OUTPUT_DIR=/output`
  - `MODEL_DIR=/workspace/artifacts/models`

Default submission-time runtime behavior:

- `PATCH_SIZE=24`
- `HYDROSAT_CALIBRATE_TEST_STATS=1`
- `HYDROSAT_TURBIDITY_MODE=model`
- `HYDROSAT_TURBIDITY_CALIBRATION=lognormal_rank`
- `HYDROSAT_TURBIDITY_LOGNORMAL_SIGMA=0.52`
- `HYDROSAT_TURBIDITY_PRIOR_SHRINK=0.05`
- `HYDROSAT_CHLA_MODE=model`
- `HYDROSAT_ENABLE_CNN=0`

The raw competition downloads are intentionally excluded from Docker with `.dockerignore`.

## Proposal Assets

- `Hydro Sat Systems_Arv Bali_baliarv21@gmail.com/hydrosat_best_technical_proposal.pptx`
- `Hydro Sat Systems_Arv Bali_baliarv21@gmail.com/presentation_script.md`
- `Hydro Sat Systems_Arv Bali_baliarv21@gmail.com/slides_sample.pptx`
- `FINAL_TECHNICAL_PROPOSAL.md`
- `README_SUBMISSION.txt`
- `SUBMISSION_MANIFEST.md`
