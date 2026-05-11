# Hydrosat Track 2 Water Quality Inversion

This project predicts two water-quality values from multi-band satellite images:

- **Turbidity**
- **Chlorophyll-a**

The pipeline uses two model families:

1. **CPU tree/boosting models** on engineered spectral features.
2. **Optional GPU CNN models** on raw 12-band image patches.

The current checked-in model artifacts are already usable for inference:

- `artifacts/models/turbidity.joblib`
- `artifacts/models/turbidity_ensemble.joblib`
- `artifacts/models/chla.joblib`
- `artifacts/models/chla_ensemble.joblib`

Current local best model selection:

- **Turbidity:** single `hgb_log` tree model, RMSE about `152.57`, R2 about `0.242`
- **Chl-a:** `xgboost + extra trees` ensemble, RMSE about `6.94`, R2 about `0.668`

CNN training code is included, but the local machine had no CUDA GPU. The GPU teammate should train the CNN candidates and send back the updated `artifacts/models` folder.

## What The GPU Teammate Should Do

Use Python 3.10 with CUDA PyTorch installed.

Check CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full GPU training pipeline.

Linux/macOS:

```bash
bash scripts/train_gpu.sh
```

Windows PowerShell:

```powershell
.\scripts\train_gpu.ps1
```

That script:

1. Checks CUDA.
2. Rebuilds feature CSVs.
3. Trains the CPU tree/boosting ensemble.
4. Trains ResNet18 and EfficientNet-B0 CNNs for turbidity.
5. Trains ResNet18 and EfficientNet-B0 CNNs for chl-a.
6. Runs sample inference.

After running, send back:

```text
artifacts/models/
artifacts/output/
```

Important: CNN inference is disabled by default for speed. To test CNN-aware inference after training:

```bash
HYDROSAT_ENABLE_CNN=1 python -m hydrosat.infer --input-root "track2_download_link_1/Guide to the Second Round_track2/test_input_sample" --model-dir artifacts/models --output-dir artifacts/output --patch-size 32
```

With `HYDROSAT_ENABLE_CNN=1`, inference still only uses a CNN if its saved validation RMSE beats the tree/ensemble model.

## Data Folders

Keep these folders with the project:

```text
track2_download_link_1/
track2_download_link_2/
track2_download_link_3/
track2_download_link_4/
track2_download_link_5/
```

They contain:

- training CSV labels
- 12-band `.tif` images
- sample test input under `track2_download_link_1/Guide to the Second Round_track2/test_input_sample`

Do not copy the dataset into Docker. `.dockerignore` excludes these folders so the image stays small.

## Main Commands

Build the standard feature table:

```bash
export PYTHONPATH=src
python -m hydrosat.build_features --data-root . --out artifacts/features/train_features.csv --patch-size 32
```

Build the second feature-table filename used by the existing training scripts:

```bash
export PYTHONPATH=src
python -m hydrosat.build_features --data-root . --out artifacts/features/train_features_v2.csv --patch-size 32
```

Train single best baseline models:

```bash
export PYTHONPATH=src
python -m hydrosat.train_baseline --features artifacts/features/train_features_v2.csv --model-dir artifacts/models_v2 --max-features 150
```

Train deployment ensembles:

```bash
export PYTHONPATH=src
python -m hydrosat.train_ensemble --features-turbidity artifacts/features/train_features_v2.csv --features-chla artifacts/features/train_features.csv --model-dir artifacts/models --top-n 2 --max-features 150 --turbidity-models hgb_log,lightgbm_log,xgboost_log,extra_log --chla-models extra,xgboost,lightgbm_log
```

Train CNNs manually:

```bash
export PYTHONPATH=src
python -m hydrosat.train_cnn --data-root . --target turbidity --model-dir artifacts/models --arch resnet18 --patch-size 64 --epochs 80 --final-epochs 100 --batch-size 64 --folds 5 --log-target --pretrained
python -m hydrosat.train_cnn --data-root . --target chla --model-dir artifacts/models --arch efficientnet_b0 --patch-size 64 --epochs 100 --final-epochs 120 --batch-size 64 --folds 5 --log-target --pretrained
```

Run sample inference:

```bash
export PYTHONPATH=src
python -m hydrosat.infer --input-root "track2_download_link_1/Guide to the Second Round_track2/test_input_sample" --model-dir artifacts/models --output-dir artifacts/output --patch-size 32 --progress-every 1000
```

Run competition/container inference:

```bash
./run.sh
```

The platform should mount test data at `/input` and set `OUTPUT_DIR`. Locally, you can override:

```bash
INPUT_DIR=/path/to/input OUTPUT_DIR=/path/to/output MODEL_DIR=artifacts/models ./run.sh
```

## Output Files

Inference writes the official Track 2 names:

```text
turbidity_result.json
chla_result.json
```

It also writes aliases because the PDF and GitLab README used different names:

```text
result_turbidity.json
result_chla.json
```

## What Each File Is For

### Root Files

`README.md`

This file. It explains the project, models, commands, teammate handoff, and file purposes.

`requirements.txt`

Python dependencies for feature extraction, tree models, CNN training, and inference:

- `numpy`
- `pandas`
- `scikit-learn`
- `joblib`
- `rasterio`
- `lightgbm`
- `xgboost`
- `catboost`

PyTorch is expected from the CUDA environment or competition base image.

`pyproject.toml`

Minimal Python package config. It lets the code be treated as a package under `src/hydrosat`.

`Dockerfile`

Competition Docker image. It uses the provided PyTorch CUDA competition base:

```text
10.200.99.202:15080/zero2x002/competition-base:pytorch2.5.1-cuda12.1-cudnn9
```

It installs requirements, copies source/model files, sets `PYTHONPATH`, and runs `run.sh`.

`.gitlab-ci.yml`

Competition GitLab pipeline. It builds the Docker image, pushes it, and starts the platform job. The important setting is:

```text
START_CMD: "./run.sh"
```

`.dockerignore`

Prevents huge local data and generated files from being copied into the Docker image. This keeps Docker build smaller and faster.

`.gitignore`

Keeps generated local artifacts and Python caches out of git.

`run.sh`

Container entrypoint. It runs:

```bash
python -m hydrosat.infer
```

It reads:

- `INPUT_DIR`, default `/input`
- `OUTPUT_DIR`, default `/output`
- `MODEL_DIR`, default `/app/artifacts/models`
- `HYDROSAT_ENABLE_CNN`, default `0`

### Scripts

`scripts/train_gpu.sh`

Linux/macOS one-command GPU training script. It builds features, trains tree ensembles, trains ResNet18/EfficientNet-B0 CNN candidates, and runs sample inference.

`scripts/train_gpu.ps1`

Windows PowerShell version of the same GPU training workflow.

### Source Package

`src/hydrosat/__init__.py`

Marks `hydrosat` as a Python package.

`src/hydrosat/paths.py`

Finds training CSV files, resolves image paths from CSV rows, and maps target names to CSV columns.

Key responsibilities:

- find `track2_*_train_point_area*.csv`
- detect turbidity vs chl-a CSVs
- find matching `areaX_images/filename.tif`
- support sample test layout where CSVs sit next to `area8_images`

`src/hydrosat/raster.py`

Reads GeoTIFF imagery.

Key responsibilities:

- open 12-band `.tif` files with `rasterio`
- convert Lon/Lat to pixel row/column
- extract a fixed-size patch around the point
- cache open raster datasets for faster large inference

`src/hydrosat/features.py`

Builds tabular features from image patches.

Feature groups:

- band mean/std/min/max/percentiles
- water-masked band stats
- NDWI
- MNDWI
- NDTI
- NDCI
- NDVI
- band ratios
- visible brightness
- center-window stats
- gradient/texture stats
- water fraction
- date seasonality
- normalized pixel position

`src/hydrosat/build_features.py`

CLI for converting all training CSV points into a feature table.

Outputs examples:

```text
artifacts/features/train_features.csv
artifacts/features/train_features_v2.csv
```

These files are generated and can be rebuilt. Both use the current rich feature extractor in `features.py`; the two names are kept because earlier experiments used `train_features.csv` for chl-a and `train_features_v2.csv` for turbidity.

`src/hydrosat/diagnose_features.py`

Data diagnostics CLI.

Reports:

- target distribution
- per-area label summary
- missing feature rates
- strongest Spearman correlations
- extreme high-label points

Useful for understanding turbidity outliers.

`src/hydrosat/train_baseline.py`

Trains single best tree/boosting model per target.

Models include:

- HistGradientBoosting
- ExtraTrees
- RandomForest
- LightGBM
- XGBoost
- CatBoost
- log-target variants

Supports:

- feature selection
- grouped CV by image/date
- stricter grouped CV by area with `--group-by area`

`src/hydrosat/train_ensemble.py`

Trains multiple tree/boosting models and saves the best top-N as an ensemble.

Current intended use:

- turbidity uses `train_features_v2.csv`
- chl-a uses `train_features.csv`
- predictions are averaged across selected models

`src/hydrosat/train_cnn.py`

GPU-oriented CNN training.

Supported architectures:

- `small`
- `resnet18`
- `resnet34`
- `resnet50`
- `efficientnet_b0`
- `efficientnet_b3`
- `convnext_tiny`

It adapts RGB pretrained backbones to 12-band inputs by replacing the first convolution. It also uses flip/rotation/noise augmentation, log-target training, SmoothL1 loss, early stopping, and CV RMSE tracking.

Saved CNN files:

```text
artifacts/models/turbidity_cnn_resnet18.pt
artifacts/models/turbidity_cnn_efficientnet_b0.pt
artifacts/models/turbidity_cnn.pt
artifacts/models/chla_cnn_resnet18.pt
artifacts/models/chla_cnn_efficientnet_b0.pt
artifacts/models/chla_cnn.pt
```

The generic `*_cnn.pt` file is only updated when a candidate CNN is the best CNN so far.

`src/hydrosat/infer.py`

Competition inference script.

It:

- reads `track2_turb_test_point.csv`
- reads `track2_cha_test_point.csv`
- extracts features for each test point
- loads tree/ensemble models
- optionally loads CNNs if `HYDROSAT_ENABLE_CNN=1`
- chooses CNN only if its validation RMSE is better
- writes JSON outputs

### Artifacts

`artifacts/models/`

Current model folder used by inference and Docker.

Kept files:

```text
chla.joblib
chla_ensemble.joblib
turbidity.joblib
turbidity_ensemble.joblib
```

These let the project run inference immediately even before teammate GPU training.

Generated but removed from handoff cleanup:

```text
artifacts/features/
artifacts/output/
artifacts/models_v2/
artifacts/models_cliptest/
catboost_info/
__pycache__/
```

They can be rebuilt and are not needed to understand or run the project.

## What Requires GPU

Requires or strongly benefits from GPU:

- `train_cnn.py`
- ResNet/EfficientNet/ConvNeXt CNN training
- large patch CNN training

Does not need GPU:

- feature extraction
- diagnostics
- tree/boosting training
- ensemble training
- normal inference
- JSON writing
- Docker entrypoint

## Final Handoff Checklist

Send teammate:

```text
README.md
requirements.txt
pyproject.toml
Dockerfile
.gitlab-ci.yml
.dockerignore
.gitignore
run.sh
scripts/
src/
artifacts/models/
track2_download_link_*/
```

They run:

```bash
bash scripts/train_gpu.sh
```

They send back:

```text
artifacts/models/
artifacts/output/
```

Then final step is to run sample inference, confirm JSON files exist, and submit.

## Google Colab GPU Training

Use this section if you do not have a local GPU.

Fastest option: open `Hydrosat_Colab_Train.ipynb` in Google Colab, set runtime to GPU, edit `PROJECT_DIR` if needed, and run all cells.

### 1. Prepare The Folder

Zip the whole project folder, including:

```text
src/
scripts/
artifacts/models/
track2_download_link_*/
requirements.txt
README.md
run.sh
Dockerfile
.gitlab-ci.yml
```

Upload the zip to Google Drive, for example:

```text
MyDrive/hydrosat/Hydrosat_final_round.zip
```

### 2. Start Colab

In Google Colab:

1. Open a new notebook.
2. Go to `Runtime > Change runtime type`.
3. Select `T4 GPU`, `L4 GPU`, or better if available.
4. Run the cells below.

### 3. Mount Drive And Unzip

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
mkdir -p /content/hydrosat
unzip -q "/content/drive/MyDrive/hydrosat/Hydrosat_final_round.zip" -d /content/hydrosat
```

If the zip extracts into a nested folder, move into the folder that contains `README.md`:

```bash
cd /content/hydrosat/Hydrosat\ final\ round
ls
```

If your folder name is different, adjust the `cd` path.

### 4. Verify GPU

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

If `cuda` is `False`, change the Colab runtime to GPU before training.

### 5. Run Final Training

Fast full run:

```bash
bash scripts/colab_gpu_train.sh
```

This trains:

- CPU tree/boosting ensemble
- ResNet18 turbidity CNN
- EfficientNet-B0 turbidity CNN
- ResNet18 chl-a CNN
- EfficientNet-B0 chl-a CNN

The CNN trainer saves each candidate separately and keeps the best CNN as:

```text
artifacts/models/turbidity_cnn.pt
artifacts/models/chla_cnn.pt
```

Inference only uses CNNs when `HYDROSAT_ENABLE_CNN=1`, and even then it only uses them if their validation RMSE beats the tree/ensemble model.

### 6. Optional Heavier CNN Pass

If Colab gives you an L4/A100 or enough time, try these after the fast run:

```bash
export PYTHONPATH=src
python -m hydrosat.train_cnn --data-root . --target turbidity --model-dir artifacts/models --arch efficientnet_b3 --patch-size 64 --epochs 80 --final-epochs 100 --batch-size 32 --folds 5 --log-target --pretrained
python -m hydrosat.train_cnn --data-root . --target chla --model-dir artifacts/models --arch efficientnet_b3 --patch-size 64 --epochs 100 --final-epochs 120 --batch-size 32 --folds 5 --log-target --pretrained
```

For even heavier testing:

```bash
export PYTHONPATH=src
python -m hydrosat.train_cnn --data-root . --target turbidity --model-dir artifacts/models --arch convnext_tiny --patch-size 64 --epochs 80 --final-epochs 100 --batch-size 32 --folds 5 --log-target --pretrained
python -m hydrosat.train_cnn --data-root . --target chla --model-dir artifacts/models --arch convnext_tiny --patch-size 64 --epochs 100 --final-epochs 120 --batch-size 32 --folds 5 --log-target --pretrained
```

If Colab runs out of memory, reduce `--batch-size` to `16`.

### 7. Run Sample Inference In Colab

```bash
export PYTHONPATH=src
HYDROSAT_ENABLE_CNN=1 python -m hydrosat.infer \
  --input-root "track2_download_link_1/Guide to the Second Round_track2/test_input_sample" \
  --model-dir artifacts/models \
  --output-dir artifacts/output \
  --patch-size 32 \
  --progress-every 1000
```

Expected output files:

```text
artifacts/output/turbidity_result.json
artifacts/output/chla_result.json
artifacts/output/result_turbidity.json
artifacts/output/result_chla.json
```

### 8. Save Models Back To Drive

```bash
zip -qr /content/hydrosat_models.zip artifacts/models artifacts/output
cp /content/hydrosat_models.zip "/content/drive/MyDrive/hydrosat/hydrosat_models.zip"
```

Download `hydrosat_models.zip` from Drive and copy `artifacts/models` back into this project.

### 9. Bulk/Real Inference In Colab

If you have a real test input folder in Drive:

```bash
export PYTHONPATH=src
HYDROSAT_ENABLE_CNN=1 python -m hydrosat.infer \
  --input-root "/content/drive/MyDrive/hydrosat/real_input" \
  --model-dir artifacts/models \
  --output-dir "/content/drive/MyDrive/hydrosat/final_output" \
  --patch-size 32 \
  --progress-every 1000
```

For 40k points, keep `--progress-every 1000` so you can see progress.
