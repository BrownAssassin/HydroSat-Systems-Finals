# HydroSat-Systems-Finals

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Task](https://img.shields.io/badge/Task-Water%20Quality%20Regression-2E8B57)](https://en.wikipedia.org/wiki/Regression_analysis)
[![Track](https://img.shields.io/badge/Track-2%20Clean%20Water-0099CC)](https://spaceaichallenge.zero2x.org/competition)
[![Runtime](https://img.shields.io/badge/Runtime-Tabular%20Ensemble-1E88E5)](https://scikit-learn.org/)
[![Project](https://img.shields.io/badge/Project-ITU%20Ingenuity%20Cup%202026-orange)](https://github.com/BrownAssassin/HydroSat-Systems-Finals)
[![Released Area8](https://img.shields.io/badge/Released%20Area8-14.58-brightgreen)](https://github.com/BrownAssassin/HydroSat-Systems-Finals)
[![Repo Size](https://img.shields.io/github/repo-size/BrownAssassin/HydroSat-Systems-Finals)](https://github.com/BrownAssassin/HydroSat-Systems-Finals)

Final-round repository for the **ITU Ingenuity Cup: AI and Space Computing Challenge** submission by **HydroSat Systems**.

This repository is focused on **Track 2: Space Intelligence Promoting Water Quality**. The final-round task is point-level prediction of:

- `turbidity`
- `chlorophyll-a`

from multispectral GeoTIFF imagery. The final-round scoring emphasizes regression quality rather than segmentation quality, and our final public-facing repository keeps the frozen inference bundle, the full score-push training path, and the supporting proposal assets together in one place.

## Team

- **Team name:** HydroSat Systems
- **Team leader:** [Arv Bali](https://github.com/ArvBali2101)
- **Team members:**
  - [Arv Bali](https://github.com/ArvBali2101)
  - [Mrinank Sivakumar](https://github.com/BrownAssassin)

## Final Frozen Result

- **Winning runtime family:** patch-based tabular ensemble
- **Winning search path:** `patch24_filter15_top3_feat800`
- **Frozen patch size:** `24x24`
- **Turbidity runtime mode:** `model`
- **Turbidity calibration:** `lognormal_rank`
- **Turbidity sigma:** `0.52`
- **Turbidity prior shrink:** `0.05`
- **Chl-a runtime mode:** `model`
- **Critical-path CNN usage:** disabled

Final released-Area8 offline evaluation:

- **Turbidity:** `RMSE = 2.0728`, `R2 = 0.1733`, `NRMSE = 0.9609`, `score = 10.6170`
- **Chl-a:** `RMSE = 1.1465`, `R2 = 0.0802`, `NRMSE = 0.7095`, `score = 18.5354`
- **Algorithm score:** `14.5762`

Locked public references:

- `docs/results/released_area8_scores.json`
- `docs/results/released_area8_scores.md`
- `docs/results/final_score_push_summary.json`
- `docs/results/final_score_push_summary.md`

Frozen runtime artifacts:

- `artifacts/models/turbidity_ensemble.joblib`
- `artifacts/models/chla_ensemble.joblib`
- `artifacts/models/runtime_env_defaults.json`

## What Is Versioned

This repo intentionally keeps the minimum Git-tracked assets needed to reproduce, package, and explain the final result:

- all code for feature extraction, training, tuning, inference, and evaluation in `src/hydrosat/`
- the final frozen inference bundle in `artifacts/models/`
- stable final evaluation summaries in `docs/results/`
- proposal assets in `docs/proposal/`
- root-level competition and submission docs

This repo intentionally does **not** keep large local-only artifacts that are better regenerated on demand:

- raw organizer download folders under `track2_download_link_*/`
- local feature caches
- local experiment folders
- local temporary prediction outputs
- local evaluation work directories
- local training logs and caches
- local virtual environments

That means a clean Git clone supports both:

- a **quick frozen-runtime inference path**
- a **full-from-scratch score-push rebuild path**, as long as the local raw organizer folders are present

## Repository Layout

```text
src/hydrosat/             # inference, feature extraction, training, tuning, evaluation
artifacts/models/         # final frozen runtime bundle kept in Git
docs/proposal/            # final PPT, presentation script, and slide template
docs/results/             # tracked final score summaries
scripts/                  # proposal deck regeneration helper
track2_download_link_*/   # local raw organizer downloads, kept out of Git
```

Optional experiment utilities that are kept for roadmap work but are **not** part of the submission-critical path:

- `src/hydrosat/train_cnn.py`
- `src/hydrosat/diagnose_features.py`
- `src/hydrosat/train_regime_ensemble.py`

## Setup

This project targets **Python 3.10**. The frozen tabular inference path does **not** require a GPU or PyTorch.

### Baseline setup for CPU or GPU machines

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Generic shell:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Quick verification:

```bash
python -c "import hydrosat, sklearn; print('hydrosat ok'); print('sklearn', sklearn.__version__)"
```

Expected baseline state:

- `hydrosat` imports without `PYTHONPATH`
- `scikit-learn 1.7.2`

### Optional GPU setup for roadmap CNN experiments

Only do this if you plan to use `train_cnn.py` or other PyTorch-based experiments.

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"
```

## Quick Inference With The Frozen Models

The runtime bundle carries its own defaults in `artifacts/models/runtime_env_defaults.json`, so the standard inference CLI reproduces the winning configuration automatically.

Sample input inference:

```bash
python -m hydrosat.infer \
  --input-root "track2_download_link_1/Guide to the Second Round_track2/test_input_sample" \
  --model-dir "artifacts/models" \
  --output-dir "artifacts/output/sample_input_final" \
  --progress-every 1000
```

This writes all four required naming variants:

- `turbidity_result.json`
- `chla_result.json`
- `result_turbidity.json`
- `result_chla.json`

## Released-Area8 Evaluation

The released-Area8 evaluator is the local source of truth for the final public score.

```bash
python -m hydrosat.evaluate_released_area8 \
  --released-root "track2_download_link_1" \
  --model-dir "artifacts/models" \
  --work-dir "artifacts/eval_input/released_area8" \
  --output-dir "artifacts/output/released_area8" \
  --progress-every 1000
```

Current measured workload:

- `372` turbidity points
- `103` chl-a points
- `475` total points

Tracked frozen result copies live in:

- `docs/results/released_area8_scores.json`
- `docs/results/released_area8_scores.md`

## Full Reproduction From Scratch

The full rebuild path assumes the raw organizer folders are available locally at the repo root:

- `track2_download_link_1/`
- `track2_download_link_2/`
- `track2_download_link_3/`
- `track2_download_link_4/`
- `track2_download_link_5/`

Run the full score-push pipeline:

```bash
python -m hydrosat.score_push \
  --data-root . \
  --released-root track2_download_link_1 \
  --features-dir artifacts/features \
  --experiments-dir artifacts/experiments \
  --runtime-model-dir artifacts/models \
  --progress-every 500 \
  --random-state 42 \
  --selection-metric score
```

That command performs the full rebuild path:

- feature generation for the searched patch sizes
- Stage A structural search
- Stage B fine search
- mixed pairing evaluation
- bounded released-Area8 tuning
- late-stage fallback comparison
- final runtime freezing back into `artifacts/models/`

The final public winner of the last push was:

- `patch24_filter15_top3_feat800__patch24_filter15_top3_feat800`

### Advanced utilities

These utilities are useful when reproducing only part of the pipeline:

- `python -m hydrosat.build_features`
- `python -m hydrosat.train_ensemble`
- `python -m hydrosat.train_regime_ensemble`
- `python -m hydrosat.evaluate_released_area8`

## Proposal And Report Assets

- Final score summaries: `docs/results/`
- Final technical proposal deck: `docs/proposal/hydrosat_best_technical_proposal.pptx`
- Final presentation script: `docs/proposal/presentation_script.md`
- Competition slide template reference: `docs/proposal/slides_sample.pptx`

## Submission Notes

- `Dockerfile` uses the competition base image and mirrors the frozen runtime defaults.
- `.gitlab-ci.yml` is aligned to `./run.sh`.
- `run.sh` launches `python -m hydrosat.infer`.
- The submission-critical runtime contract remains:
  - `/input`
  - `/output`
  - `/workspace/artifacts/models`

Default frozen runtime behavior:

- `PATCH_SIZE=24`
- `HYDROSAT_CALIBRATE_TEST_STATS=1`
- `HYDROSAT_TURBIDITY_MODE=model`
- `HYDROSAT_TURBIDITY_CALIBRATION=lognormal_rank`
- `HYDROSAT_TURBIDITY_LOGNORMAL_SIGMA=0.52`
- `HYDROSAT_TURBIDITY_PRIOR_SHRINK=0.05`
- `HYDROSAT_CHLA_MODE=model`
- `HYDROSAT_ENABLE_CNN=0`

## Proposal And Report Assets

- `docs/proposal/hydrosat_best_technical_proposal.pptx`
- `docs/proposal/presentation_script.md`
- `docs/proposal/slides_sample.pptx`
- `docs/results/released_area8_scores.json`
- `docs/results/final_score_push_summary.json`
- `FINAL_TECHNICAL_PROPOSAL.md`
- `README_SUBMISSION.txt`
- `SUBMISSION_MANIFEST.md`
