# Submission Manifest

## Final Git-Tracked Submission Bundle

The final inference-only GitLab submission bundle is:

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

That is the complete container payload needed to build the Track 2 runtime image and run predictions from `/input` to `/output`.

## Frozen Runtime Artifacts

Tracked runtime artifacts:

- `artifacts/models/turbidity_ensemble.joblib`
- `artifacts/models/chla_ensemble.joblib`
- `artifacts/models/runtime_env_defaults.json`

Frozen runtime defaults:

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

Frozen released-Area8 score:

- Turbidity score: `10.6170`
- Chl-a score: `18.5354`
- Algorithm score: `14.5762`

Improvement trail:

- previous frozen runtime score: `14.4445`
- late-stage gain: `+0.1318` points (`+0.91%`)
- total gain over the earlier `12.6653` baseline: `+1.9109` points (`+15.09%`)

## Final Public Docs And Results

These files are tracked for public reference, but they are not needed inside the competition runtime image:

- `FINAL_TECHNICAL_PROPOSAL.md`
- `SUBMISSION_MANIFEST.md`
- `docs/results/released_area8_scores.json`
- `docs/results/released_area8_scores.md`
- `docs/results/final_score_push_summary.json`
- `docs/results/final_score_push_summary.md`
- `docs/results/site_adaptive_research.json`
- `docs/results/site_adaptive_research.md`
- `docs/proposal/hydrosat_best_technical_proposal.pptx`
- `docs/proposal/presentation_script.md`
- `docs/proposal/slides_sample.pptx`

## Local-Only Raw Data

These stay at the repo root for compatibility, but they are intentionally not tracked and not shipped in the submission image:

- `track2_download_link_1/`
- `track2_download_link_2/`
- `track2_download_link_3/`
- `track2_download_link_4/`
- `track2_download_link_5/`

## Local-Only Experiment Outputs

These are reproducible development artifacts, not part of the frozen submission bundle:

- feature tables under `artifacts/features/`
- experiment summaries under `artifacts/experiments/`
- evaluator working inputs under `artifacts/eval_input/`
- scratch prediction outputs under `artifacts/output/`
- regenerated local reports under `artifacts/reports/`
- local caches such as `catboost_info/`

## Public Interpretation Note

The frozen submission bundle and the post-release research layer answer different questions:

- Frozen leaderboard-comparable runtime score: `14.5762`
- Site-adaptive monitored-basin research score: `53.9463`

The second number is documented for proposal and research purposes only. It is not a replacement for the frozen runtime result.

## Output Contract

The runtime writes both naming variants required by the challenge materials:

```text
turbidity_result.json
chla_result.json
result_turbidity.json
result_chla.json
```
