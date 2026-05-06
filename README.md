# HydroSat Systems Finals

Final-round workspace for `Track 2: Space Intelligence Promoting Water Quality` in the `ITU AI and Space Computing Challenge`.

This repo now contains a working tabular baseline for the final round:

- normalized local raw data under `data/raw/`
- CRS-aware raster patch extraction from the multispectral GeoTIFFs
- patch-level spectral/statistical feature generation
- grouped-CV model selection for `turbidity` and `chla`
- saved deployable model bundles under `models/`
- real inference that writes the official Track 2 JSON outputs

## Current Baseline

Phase 2 is now implemented end to end.

The current baseline is tabular-first:

- build fixed `32x32` pixel patches around labeled points
- derive per-band, water-masked, index-based, center-window, and gradient features
- train separate regressors for `turbidity` and `chla`
- select the shipped baseline by `area`-grouped CV, with `filename`-grouped CV saved as a secondary check

The current saved baseline bundles are:

- `models/turbidity.joblib`
- `models/chla.joblib`

From the latest local run, the selected configurations were:

- `turbidity`: `lightgbm` with `test_range_weighted`
- `chla`: `hgb` with `full`

## Repo Layout

```text
data/raw/
  guides/
  sample_input/
  train/
docs/competition/
models/
artifacts/
scripts/
src/hydrosat/
tests/
```

`data/raw/` stays local and untracked. It is excluded from git and from Docker builds.

## Local Setup

Use the project-local Python `3.12` environment, not the system `3.14`.

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The baseline was validated locally with:

- `numpy`
- `pandas`
- `rasterio`
- `scikit-learn`
- `joblib`
- `lightgbm`
- `xgboost`

## Main Commands

Run the repo tests:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Build the full training feature table:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m hydrosat.features `
  --data-root data/raw `
  --out artifacts/features/train_features.csv `
  --patch-size 32 `
  --progress-every 100
```

Train both targets and save the best deployable bundles:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m hydrosat.train `
  --target all `
  --data-root data/raw `
  --features artifacts/features/train_features.csv `
  --models-dir models `
  --diagnostics-dir artifacts/diagnostics `
  --rebuild-features `
  --patch-size 32 `
  --max-features 180
```

Run local sample inference:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m hydrosat.infer `
  --input-root data/raw/sample_input `
  --output-dir artifacts/output `
  --models-dir models
```

Run the competition-style entrypoint locally from a bash-compatible shell:

```bash
INPUT_DIR=data/raw/sample_input OUTPUT_DIR=artifacts/output HYDROSAT_MODELS_DIR=models bash ./run.sh
```

Export the submission-safe subset into the official GitLab working tree:

```powershell
python scripts/export_submission.py --dest D:\path\to\track2_round2_model
```

## Output Contract

The authoritative Track 2 output files are:

- `/output/result_turbidity.json`
- `/output/result_chla.json`

Each JSON maps keys in the form:

```text
{filename}_{Lon}_{Lat}
```

to a single numeric prediction inside a list.

Example:

```json
{
  "area8_2024-01-15.tif_-122.823396_44.498493": [14.63]
}
```

## Why Area-Grouped CV

The hidden competition test is an unseen geographic region (`area8`), so `area`-grouped CV is the primary selection signal for the baseline. It is a better proxy for leaderboard behavior than random row splits or same-image validation.

`filename`-grouped CV is still saved as a secondary diagnostic so we can compare within-distribution stability against the harder geographic holdout.

## Notes

- When the generic submission manual conflicts with the Track 2 task PDF, follow the Track 2 task PDF for output naming.
- The baseline includes a hidden-test-aware weighted training variant because the organizer-released hidden-label statistics are much lower-tailed than the public training labels.
- The root `.gitlab-ci.yml` remains a local scaffold. Before submitting, merge its paths and command settings into the official competition GitLab template.
