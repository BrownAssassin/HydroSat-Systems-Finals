HydroSat Systems - Track 2 submission package

This repository is prepared as a self-contained competition submission. It does not download code from external repositories during runtime.

Runtime contract:
- working directory: /workspace
- input directory: /input
- output directory: /output
- model directory: /workspace/artifacts/models

Entrypoint:
- ./run.sh
- run.sh executes: python -m hydrosat.infer

Output files written by inference:
- turbidity_result.json
- chla_result.json
- result_turbidity.json
- result_chla.json

Current checked-in runtime model artifacts:
- artifacts/models/turbidity.joblib
- artifacts/models/turbidity_ensemble.joblib
- artifacts/models/chla.joblib
- artifacts/models/chla_ensemble.joblib
- artifacts/models/turbidity_cnn.pt
- artifacts/models/chla_cnn.pt

Default behavior:
- ensemble tree/boosting inference is the primary path
- CNN inference is optional and disabled by default
- released-test calibration is enabled by default in run.sh

Released Area8 offline evaluation using the official final-round scoring formula:
- Turbidity: RMSE 2.4604, R2 -0.1649, score 0.0000
- Chl-a: RMSE 1.2252, R2 -0.0503, score 12.0906
- Algorithm score: 6.0453

Competition-facing files at the repo root:
- .gitlab-ci.yml
- Dockerfile
- requirements.txt
- pyproject.toml
- run.sh

Code and model directories needed for submission:
- src/
- artifacts/models/

Do not include these in the Docker context or GitLab submission bundle:
- track2_download_link_*/
- artifacts/output/
- artifacts/reports/
- artifacts/eval_input/
- local virtual environments and caches
