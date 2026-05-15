HydroSat Systems - Track 2 inference submission bundle

This repository is packaged as a self-contained final-round inference submission for the ITU Ingenuity Cup 2026, Track 2: Space Intelligence Promoting Water Quality.

Submission runtime contract:
- working directory: `/workspace`
- input directory: `/input`
- output directory: `/output`
- model directory: `/workspace/artifacts/models`
- entrypoint: `./run.sh`

Frozen runtime bundle:
- `artifacts/models/turbidity_ensemble.joblib`
- `artifacts/models/chla_ensemble.joblib`
- `artifacts/models/runtime_env_defaults.json`

Frozen default behavior:
- patch size `24`
- turbidity mode `model`
- turbidity calibration `lognormal_rank`
- turbidity lognormal sigma `0.52`
- turbidity prior shrink `0.05`
- chl-a mode `model`
- released-stat calibration enabled
- CNN disabled on the critical path

Track 2 output files:
- `turbidity_result.json`
- `chla_result.json`
- `result_turbidity.json`
- `result_chla.json`

Final released-Area8 offline evidence:
- Turbidity: `RMSE 2.0728`, `R2 0.1733`, `score 10.6170`
- Chl-a: `RMSE 1.1465`, `R2 0.0802`, `score 18.5354`
- Algorithm score: `14.5762`

Final submission-facing files:
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

Do not include local-only assets in the GitLab submission image:
- `track2_download_link_*/`
- local experiment outputs under `artifacts/`
- `docs/proposal/`
- `docs/results/`
- local virtual environments and caches

Stable public references:
- score summaries: `docs/results/`
- proposal assets: `docs/proposal/`

Public interpretation note:
- leaderboard-comparable frozen runtime score: `14.5762`
- separate site-adaptive research score: `53.9463`
- the site-adaptive result is a post-release retrospective monitored-site study, not a replacement leaderboard score
