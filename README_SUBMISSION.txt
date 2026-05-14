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

Frozen runtime defaults:
- released-stat calibration enabled
- turbidity mode `blend`
- turbidity heuristic weight `0.81`
- chl-a mode `model`
- CNNs disabled

Final released Area8 offline evaluation using the official final-round scoring formula:
- Turbidity: `RMSE 2.1440`, `R2 0.1155`, `score 6.0765`
- Chl-a: `RMSE 1.1400`, `R2 0.0906`, `score 19.2541`
- Algorithm score: `12.6653`

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
