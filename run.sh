#!/usr/bin/env bash
set -euo pipefail

export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HYDROSAT_ENABLE_CNN="${HYDROSAT_ENABLE_CNN:-0}"
export HYDROSAT_CALIBRATE_TEST_STATS="${HYDROSAT_CALIBRATE_TEST_STATS:-1}"
export HYDROSAT_CALIBRATE_TURBIDITY_TEST_STATS="${HYDROSAT_CALIBRATE_TURBIDITY_TEST_STATS:-1}"
export HYDROSAT_CALIBRATE_CHLA_TEST_STATS="${HYDROSAT_CALIBRATE_CHLA_TEST_STATS:-1}"
export HYDROSAT_TURBIDITY_PRIOR_SHRINK="${HYDROSAT_TURBIDITY_PRIOR_SHRINK:-0.45}"
export HYDROSAT_CHLA_PRIOR_SHRINK="${HYDROSAT_CHLA_PRIOR_SHRINK:-0}"
export HYDROSAT_NEUTRALIZE_GEO="${HYDROSAT_NEUTRALIZE_GEO:-0}"
export HYDROSAT_TURBIDITY_MODE="${HYDROSAT_TURBIDITY_MODE:-blend}"
export HYDROSAT_TURBIDITY_INVERT_RANK="${HYDROSAT_TURBIDITY_INVERT_RANK:-0}"
export HYDROSAT_TURBIDITY_HEURISTIC_WEIGHT="${HYDROSAT_TURBIDITY_HEURISTIC_WEIGHT:-0.45}"
export HYDROSAT_TURBIDITY_CALIBRATION="${HYDROSAT_TURBIDITY_CALIBRATION:-lognormal_rank}"
export HYDROSAT_TURBIDITY_LOGNORMAL_SIGMA="${HYDROSAT_TURBIDITY_LOGNORMAL_SIGMA:-0.80}"
export HYDROSAT_CHLA_MODE="${HYDROSAT_CHLA_MODE:-model}"
export HYDROSAT_CHLA_HEURISTIC_WEIGHT="${HYDROSAT_CHLA_HEURISTIC_WEIGHT:-0.25}"
export HYDROSAT_CHLA_INVERT_RANK="${HYDROSAT_CHLA_INVERT_RANK:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SRC_DIR="/workspace/src"
if [ -d "${SCRIPT_DIR}/src" ]; then
  DEFAULT_SRC_DIR="${SCRIPT_DIR}/src"
  if command -v cygpath >/dev/null 2>&1; then
    DEFAULT_SRC_DIR="$(cygpath -w "${SCRIPT_DIR}/src")"
  fi
fi
export HYDROSAT_SRC_DIR="${HYDROSAT_SRC_DIR:-${DEFAULT_SRC_DIR}}"
export PYTHONPATH="${PYTHONPATH:-${HYDROSAT_SRC_DIR}}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
  if [ -x ".venv/Scripts/python.exe" ]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
  else
    PYTHON_BIN="python"
  fi
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [ ! -x "${PYTHON_BIN}" ]; then
  if command -v py >/dev/null 2>&1; then
    PYTHON_BIN="py"
  elif command -v py.exe >/dev/null 2>&1; then
    PYTHON_BIN="py.exe"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No usable Python interpreter found in PATH" >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" -c "import os, runpy, sys; sys.path.insert(0, os.environ['HYDROSAT_SRC_DIR']); runpy.run_module('hydrosat.infer', run_name='__main__')" --input-root "${INPUT_DIR:-/input}" --output-dir "${OUTPUT_DIR:-/output}" --model-dir "${MODEL_DIR:-/workspace/artifacts/models}" --patch-size "${PATCH_SIZE:-32}" --progress-every "${PROGRESS_EVERY:-1000}"
