HydroSat Systems - Track 2 final submission package

This repository is prepared as a self-contained competition submission. The container does not download code or models from external repositories during runtime.

Runtime contract:
- working directory: `/workspace`
- input directory: `/input`
- output directory: `/output`
- model directory: `/workspace/artifacts/models`

Entrypoint:
- `./run.sh`
- `run.sh` executes `python -m hydrosat.infer`

Output files written by inference:
- `turbidity_result.json`
- `chla_result.json`
- `result_turbidity.json`
- `result_chla.json`

Final frozen runtime model artifacts:
- `artifacts/models/turbidity_ensemble.joblib`
- `artifacts/models/chla_ensemble.joblib`
- `artifacts/models/runtime_env_defaults.json`

Frozen runtime defaults:
- patch size `24`
- released-stat calibration enabled
- turbidity mode `model`
- turbidity calibration `lognormal_rank`
- turbidity lognormal sigma `0.52`
- turbidity prior shrink `0.05`
- chl-a mode `model`
- CNNs disabled

Final released Area8 offline evaluation using the official final-round scoring formula:
- Turbidity: `RMSE 2.0728`, `R2 0.1733`, `score 10.6170`
- Chl-a: `RMSE 1.1465`, `R2 0.0802`, `score 18.5354`
- Algorithm score: `14.5762`

Latest late-stage improvement:
- previous frozen runtime score `14.4445`
- gain `+0.1318` points

Competition-facing files at the repo root:
- `.gitlab-ci.yml`
- `Dockerfile`
- `requirements.txt`
- `pyproject.toml`
- `run.sh`

Code and model directories needed for submission:
- `src/`
- `artifacts/models/`

Do not include these in the GitLab submission image:
- `track2_download_link_*/`
- `artifacts/eval_input/`
- `artifacts/output/`
- `artifacts/reports/`
- `artifacts/features/`
- `artifacts/experiments/`
- local virtual environments and caches
