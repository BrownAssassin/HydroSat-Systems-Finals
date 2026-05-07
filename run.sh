#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
HYDROSAT_DATA_ROOT="${HYDROSAT_DATA_ROOT:-data/raw}"
HYDROSAT_MODELS_DIR="${HYDROSAT_MODELS_DIR:-${MODEL_DIR:-models}}"
PYTHONPATH="${PYTHONPATH:-src}"

mkdir -p "${OUTPUT_DIR}"

export INPUT_DIR
export OUTPUT_DIR
export HYDROSAT_DATA_ROOT
export HYDROSAT_MODELS_DIR
export PYTHONPATH

echo "HydroSat final-round scaffold"
echo "INPUT_DIR=${INPUT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "HYDROSAT_DATA_ROOT=${HYDROSAT_DATA_ROOT}"
echo "HYDROSAT_MODELS_DIR=${HYDROSAT_MODELS_DIR}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "INPUT_DIR contents:"
find "${INPUT_DIR}" -maxdepth 2 -type f 2>/dev/null | sort | head -n 50 || true

python -m hydrosat.infer "$@"
