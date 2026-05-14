# Submission Manifest

## Final Competition Submission Bundle

The final runnable GitLab submission bundle should contain:

- `.gitattributes`
- `.gitlab-ci.yml`
- `.dockerignore`
- `Dockerfile`
- `README.md`
- `README_SUBMISSION.txt`
- `requirements.txt`
- `pyproject.toml`
- `run.sh`
- `src/`
- `artifacts/models/`

That is the complete inference-only container payload needed to build the image and run Track 2 predictions on `/input`.

## Frozen Runtime Contents

Frozen runtime model files:

- `artifacts/models/turbidity_ensemble.joblib`
- `artifacts/models/chla_ensemble.joblib`

Frozen runtime footprint:

- `29.48 MB`

Frozen released Area8 score:

- Turbidity score: `6.0765`
- Chl-a score: `19.2541`
- Algorithm score: `12.6653`

## Local-Only Repo Assets

These stay in the development workspace but are **not** part of the submission image:

- `track2_download_link_1/` to `track2_download_link_5/`
- `artifacts/eval_input/`
- `artifacts/output/`
- `artifacts/reports/`
- `artifacts/features/`
- `artifacts/experiments/`
- `Hydro Sat Systems_Arv Bali_baliarv21@gmail.com/`
- `FINAL_TECHNICAL_PROPOSAL.md`
- `scripts/build_proposal_deck.py`

## Runtime Contract

Entrypoint:

```text
./run.sh
```

Primary runtime model location:

```text
artifacts/models/
```

Default runtime behavior:

```text
HYDROSAT_CALIBRATE_TEST_STATS=1
HYDROSAT_TURBIDITY_MODE=blend
HYDROSAT_TURBIDITY_HEURISTIC_WEIGHT=0.81
HYDROSAT_CHLA_MODE=model
HYDROSAT_ENABLE_CNN=0
```

## Output Contract

The package writes both naming variants required by the challenge materials:

```text
turbidity_result.json
chla_result.json
result_turbidity.json
result_chla.json
```
