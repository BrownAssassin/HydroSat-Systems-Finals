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

Frozen runtime files:

- `artifacts/models/turbidity_ensemble.joblib`
- `artifacts/models/chla_ensemble.joblib`
- `artifacts/models/runtime_env_defaults.json`

Frozen runtime footprint:

- `4.84 MB`

Frozen released Area8 score:

- Turbidity score: `10.6170`
- Chl-a score: `18.5354`
- Algorithm score: `14.5762`

Improvement over the prior frozen baseline:

- previous frozen runtime score: `14.4445`
- late-stage gain: `+0.1318` points (`+0.91%`)
- total gain over the earlier `12.6653` baseline: `+1.9109` points (`+15.09%`)

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
PATCH_SIZE=24
HYDROSAT_CALIBRATE_TEST_STATS=1
HYDROSAT_TURBIDITY_MODE=model
HYDROSAT_TURBIDITY_CALIBRATION=lognormal_rank
HYDROSAT_TURBIDITY_LOGNORMAL_SIGMA=0.52
HYDROSAT_TURBIDITY_PRIOR_SHRINK=0.05
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
