# HydroSat Systems - Track 2 Final Round

HydroSat Systems is our final-round repository for `Track 2: Space Intelligence Promoting Water Quality` in the ITU AI and Space Computing Challenge. The frozen submission path is a CPU-first water-quality inference pipeline that predicts:

- turbidity
- chlorophyll-a

from 12-band GeoTIFF imagery using point-centered patch extraction, handcrafted spectral-spatial features, and target-specific ensemble regressors.

## Final Frozen Runtime

The current default runtime is the best validated score-push candidate we produced locally before the deadline:

- feature patch size: `32x32`
- training selection mode: `area`-grouped ensemble CV
- selected experiment: `area_filter10_score`
- turbidity runtime mode: `blend`
- turbidity blend weight: `0.81`
- released-stat calibration: enabled
- CNNs: disabled on the critical path

Runtime model files:

- `artifacts/models/turbidity_ensemble.joblib`
- `artifacts/models/chla_ensemble.joblib`

Runtime artifact footprint:

- frozen submission bundle under `artifacts/models/`: `29.48 MB`

## Final Released Area8 Offline Evaluation

Using the official released Area8 truth JSONs and the final-round scoring formula:

- Turbidity: `RMSE = 2.1440`, `R2 = 0.1155`, `NRMSE = 0.9940`, `score = 6.0765`
- Chl-a: `RMSE = 1.1400`, `R2 = 0.0906`, `NRMSE = 0.7055`, `score = 19.2541`
- Algorithm score: `12.6653`

Those final numbers are written to:

- `artifacts/reports/released_area8/released_area8_scores.json`
- `artifacts/reports/released_area8/released_area8_scores.md`

The score-push experiment table is written to:

- `artifacts/reports/score_push_experiments.json`
- `artifacts/reports/score_push_experiments.md`

## Repo Layout

- `src/hydrosat/` contains feature extraction, raster IO, training CLIs, inference CLI, released-Area8 scoring, and the score-push sweep runner.
- `artifacts/models/` contains the frozen runtime models used by inference.
- `track2_download_link_1/` to `track2_download_link_5/` contain the raw organizer downloads and stay at the repo root for compatibility.
- `Hydro Sat Systems_Arv Bali_baliarv21@gmail.com/` contains the final deck and presentation script.
- `scripts/build_proposal_deck.py` regenerates the PPT from the current score report.

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

```powershell
$env:PYTHONPATH = "src"
python -m hydrosat.infer `
  --input-root "track2_download_link_1\Guide to the Second Round_track2\test_input_sample" `
  --model-dir "artifacts\models" `
  --output-dir "artifacts\output\sample_input" `
  --patch-size 32 `
  --progress-every 1000
```

This writes all four naming variants:

- `turbidity_result.json`
- `chla_result.json`
- `result_turbidity.json`
- `result_chla.json`

## Re-Run Final Released Area8 Evaluation

```powershell
$env:PYTHONPATH = "src"
python -m hydrosat.evaluate_released_area8 `
  --released-root "track2_download_link_1" `
  --model-dir "artifacts\models" `
  --work-dir "artifacts\eval_input\released_area8" `
  --output-dir "artifacts\output\released_area8" `
  --report-dir "artifacts\reports\released_area8" `
  --patch-size 32 `
  --progress-every 500
```

Current measured workload:

- `372` turbidity points
- `103` chl-a points
- `475` total points

## Reproduce The Final Score Push

Build features once:

```powershell
$env:PYTHONPATH = "src"
python -m hydrosat.build_features --data-root . --out artifacts/features/train_features_area32.csv --patch-size 32 --progress-every 200
Copy-Item artifacts/features/train_features_area32.csv artifacts/features/train_features.csv -Force
Copy-Item artifacts/features/train_features_area32.csv artifacts/features/train_features_v2.csv -Force
```

Run the fixed tabular sweep:

```powershell
$env:PYTHONPATH = "src"
python -m hydrosat.score_push `
  --data-root . `
  --released-root track2_download_link_1 `
  --features-dir artifacts/features `
  --experiments-dir artifacts/experiments `
  --reports-dir artifacts/reports `
  --runtime-model-dir artifacts/models `
  --patch-size 32 `
  --progress-every 200 `
  --top-n 5 `
  --max-features 500 `
  --selection-metric score
```

The winning sweep result before final freezing was:

- `area_filter10_score`
- area-CV algorithm score proxy: `11.6771`
- image-CV algorithm score proxy: `56.3578`

After final runtime calibration defaults were frozen, the released Area8 score rose to `12.6653`.

## Submission Notes

- `Dockerfile` uses the competition PyTorch CUDA base image.
- `.gitlab-ci.yml` is aligned to `./run.sh`.
- `run.sh` expects:
  - `INPUT_DIR=/input`
  - `OUTPUT_DIR=/output`
  - `MODEL_DIR=/workspace/artifacts/models`

Default submission-time runtime behavior:

- `HYDROSAT_CALIBRATE_TEST_STATS=1`
- `HYDROSAT_TURBIDITY_MODE=blend`
- `HYDROSAT_TURBIDITY_HEURISTIC_WEIGHT=0.81`
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
