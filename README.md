# HydroSat Systems - Track 2 Final Round

HydroSat Systems is our final-round Track 2 repository for the ITU AI and Space Computing Challenge. The checked-in baseline is a CPU-first water-quality inference package that predicts:

- turbidity
- chlorophyll-a

from 12-band GeoTIFF imagery using engineered spectral-spatial features and target-specific ensemble regressors. Optional CNN artifacts are present, but the default runnable path remains the tree/boosting ensemble stack.

## Current Baseline Status

The current repo has been audited against the imported teammate handoff and cleaned around the files that are actually runnable today:

- Python package: `src/hydrosat/`
- deployed model artifacts: `artifacts/models/`
- competition entrypoint: `run.sh`
- competition container files: `Dockerfile` and `.gitlab-ci.yml`
- raw organizer downloads kept at the repo root: `track2_download_link_*`

The old README references to missing helper scripts and missing Area8 scenes have been removed. The full released Area8 image bundle is now present under `track2_download_link_1/area8_images`.

## Released Area8 Offline Evaluation

Using the official released Area8 truth JSONs and the scoring formula from the final-round task PDF:

- Turbidity: `RMSE = 2.4604`, `R2 = -0.1649`, `NRMSE = 1.1407`, `score = 0.0000`
- Chl-a: `RMSE = 1.2252`, `R2 = -0.0503`, `NRMSE = 0.7582`, `score = 12.0906`
- Algorithm score: `6.0453`

Those numbers are generated locally by `python -m hydrosat.evaluate_released_area8` and saved to:

- `artifacts/reports/released_area8/released_area8_scores.json`
- `artifacts/reports/released_area8/released_area8_scores.md`

## Repo Layout

- `src/hydrosat/` contains feature extraction, raster IO, training CLIs, inference CLI, and the released-Area8 evaluator.
- `artifacts/models/` contains the current runtime model artifacts used by inference.
- `track2_download_link_1/` to `track2_download_link_5/` contain the raw competition downloads and are left in place for compatibility.
- `Hydro Sat Systems_Arv Bali_baliarv21@gmail.com/` contains the proposal deck assets and the presentation script.

Generated local outputs are written under `artifacts/` and ignored by git:

- `artifacts/eval_input/`
- `artifacts/output/`
- `artifacts/reports/`

## Local Setup On This Windows GPU Machine

Use Python `3.10`.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify the GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected on this machine:

- CUDA available: `True`
- device: `NVIDIA GeForce RTX 5070 Laptop GPU`

## Run Sample Input Inference

```powershell
$env:PYTHONPATH = "src"
python -m hydrosat.infer `
  --input-root "track2_download_link_1\Guide to the Second Round_track2\test_input_sample" `
  --model-dir "artifacts\models" `
  --output-dir "artifacts\output\sample_input" `
  --patch-size 32 `
  --progress-every 1000
```

This writes all four output names:

- `turbidity_result.json`
- `chla_result.json`
- `result_turbidity.json`
- `result_chla.json`

## Run Released Area8 Offline Evaluation

The released truth files do not ship with the full test-point CSVs, so the evaluator reconstructs them from the official truth JSON keys into an ignored working directory and then scores the predictions with the official formula:

- `NRMSE = RMSE / mean_truth`
- `parameter_score = (0.5 * max(0, R2) + 0.5 * max(0, 1 - NRMSE)) * 100`
- `algorithm_score = 0.5 * turbidity_score + 0.5 * chla_score`

Run it with:

```powershell
$env:PYTHONPATH = "src"
python -m hydrosat.evaluate_released_area8 `
  --released-root "track2_download_link_1" `
  --model-dir "artifacts\models" `
  --work-dir "artifacts\eval_input\released_area8" `
  --output-dir "artifacts\output\released_area8" `
  --report-dir "artifacts\reports\released_area8" `
  --patch-size 32 `
  --progress-every 100
```

Local runtime on this machine for the full released Area8 set:

- `372` turbidity points
- `103` chl-a points
- `475` total points
- about `20.43` seconds end to end

## Current Runtime Artifacts

Checked-in model files:

- `artifacts/models/turbidity.joblib`
- `artifacts/models/turbidity_ensemble.joblib`
- `artifacts/models/chla.joblib`
- `artifacts/models/chla_ensemble.joblib`
- `artifacts/models/turbidity_cnn.pt`
- `artifacts/models/chla_cnn.pt`

Footprint:

- default ensemble inference bundle: about `29.57 MB`
- full checked-in model artifact folder: about `73.01 MB`

The default inference path loads the ensemble `.joblib` files first. CNNs are optional and disabled by default.

## Optional Retraining Commands

Retraining is optional and was not required for the released Area8 evaluation above. If we decide to tune further later, the direct CLIs currently present are:

Build features:

```powershell
$env:PYTHONPATH = "src"
python -m hydrosat.build_features --data-root . --out artifacts/features/train_features.csv --patch-size 32
python -m hydrosat.build_features --data-root . --out artifacts/features/train_features_v2.csv --patch-size 32
```

Train baseline trees:

```powershell
$env:PYTHONPATH = "src"
python -m hydrosat.train_baseline --features artifacts/features/train_features_v2.csv --model-dir artifacts/models_v2 --max-features 150
```

Train deployment ensembles:

```powershell
$env:PYTHONPATH = "src"
python -m hydrosat.train_ensemble `
  --features-turbidity artifacts/features/train_features_v2.csv `
  --features-chla artifacts/features/train_features.csv `
  --model-dir artifacts/models `
  --top-n 2 `
  --max-features 150 `
  --turbidity-models hgb_log,lightgbm_log,xgboost_log,extra_log `
  --chla-models extra,xgboost,lightgbm_log
```

Train CNNs:

```powershell
$env:PYTHONPATH = "src"
python -m hydrosat.train_cnn --data-root . --target turbidity --model-dir artifacts/models --arch resnet18 --patch-size 64 --epochs 80 --final-epochs 100 --batch-size 64 --folds 5 --log-target --pretrained
python -m hydrosat.train_cnn --data-root . --target chla --model-dir artifacts/models --arch efficientnet_b0 --patch-size 64 --epochs 100 --final-epochs 120 --batch-size 64 --folds 5 --log-target --pretrained
```

## Submission Notes

- `Dockerfile` uses the competition PyTorch CUDA base image.
- `.gitlab-ci.yml` is restored and aligned to `./run.sh`.
- `run.sh` expects:
  - `INPUT_DIR=/input`
  - `OUTPUT_DIR=/output`
  - `MODEL_DIR=/workspace/artifacts/models`

The raw competition downloads are intentionally not copied into Docker. `.dockerignore` excludes the `track2_download_link_*` folders and local generated outputs.

## Proposal Assets

Proposal assets live here:

- `Hydro Sat Systems_Arv Bali_baliarv21@gmail.com/hydrosat_best_technical_proposal.pptx`
- `Hydro Sat Systems_Arv Bali_baliarv21@gmail.com/slides_sample.pptx`
- `Hydro Sat Systems_Arv Bali_baliarv21@gmail.com/presentation_script.md`
- `scripts/build_proposal_deck.py`

Supporting written materials:

- `FINAL_TECHNICAL_PROPOSAL.md`
- `README_SUBMISSION.txt`
- `SUBMISSION_MANIFEST.md`
