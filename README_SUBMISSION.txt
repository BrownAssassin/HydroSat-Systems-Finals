HydroSat Systems - Clean Water Track final model package

This package is self-contained and does not download source code from GitHub.

Expected runtime paths:
- Input directory: /input or INPUT_DIR
- Output directory: /output or OUTPUT_DIR
- Model directory: /app/artifacts/models or MODEL_DIR

Docker usage:
docker build -t hydrosat-final .
docker run --rm -v /path/to/input:/input -v /path/to/output:/output hydrosat-final

The image runs ./run.sh, which calls:
python -m hydrosat.infer

It writes:
- result_turbidity.json
- turbidity_result.json
- result_chla.json
- chla_result.json

The included configuration uses the successful d13ff94 turbidity pipeline
with public-stat lognormal rank calibration, plus the Kaggle
distribution-filtered CHLA ensemble.

Local released-label score estimate after the final calibration sweep:
- Turbidity: RMSE 2.0490, R2 0.1998, score 13.1550
- Chl-a: RMSE 1.1551, R2 0.0664, score 17.5815
- Algorithm score: 15.3682

Important runtime calibration defaults:
- HYDROSAT_TURBIDITY_HEURISTIC_WEIGHT=0.45
- HYDROSAT_TURBIDITY_LOGNORMAL_SIGMA=0.80
- HYDROSAT_TURBIDITY_PRIOR_SHRINK=0.45
- HYDROSAT_CHLA_MODE=model
