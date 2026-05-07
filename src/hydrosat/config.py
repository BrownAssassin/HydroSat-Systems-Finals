"""Shared configuration constants for the HydroSat final-round baseline."""

from __future__ import annotations

from pathlib import Path

DEFAULT_DATA_ROOT = Path("data/raw")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_INPUT_DIR = Path("/input")
DEFAULT_OUTPUT_DIR = Path("/output")
DEFAULT_FEATURES_PATH = Path("artifacts/features/train_features.csv")
DEFAULT_DIAGNOSTICS_DIR = Path("artifacts/diagnostics")
DEFAULT_PATCH_SIZE = 32

TRAIN_ROOT_NAME = "train"
SAMPLE_INPUT_ROOT_NAME = "sample_input"
GUIDES_ROOT_NAME = "guides"

TRAIN_AREAS = ("area1", "area2", "area3", "area5", "area6", "area7")

TURBIDITY_TARGET = "turbidity"
CHLA_TARGET = "chla"
ALL_TARGETS = (TURBIDITY_TARGET, CHLA_TARGET)

TURBIDITY_RESULT_FILENAME = "result_turbidity.json"
CHLA_RESULT_FILENAME = "result_chla.json"
REQUIRED_OUTPUT_FILENAMES = (TURBIDITY_RESULT_FILENAME, CHLA_RESULT_FILENAME)
TURBIDITY_RESULT_ALIAS_FILENAME = "turbidity_result.json"
CHLA_RESULT_ALIAS_FILENAME = "chla_result.json"
OUTPUT_ALIAS_FILENAMES = {
    TURBIDITY_TARGET: (TURBIDITY_RESULT_ALIAS_FILENAME,),
    CHLA_TARGET: (CHLA_RESULT_ALIAS_FILENAME,),
}

TARGET_RESULT_FILENAMES = {
    TURBIDITY_TARGET: TURBIDITY_RESULT_FILENAME,
    CHLA_TARGET: CHLA_RESULT_FILENAME,
}

TARGET_POINT_FILENAMES = {
    TURBIDITY_TARGET: "track2_turb_test_point.csv",
    CHLA_TARGET: "track2_cha_test_point.csv",
}

TARGET_TRAIN_FILENAMES = {
    TURBIDITY_TARGET: "track2_turb_train_point_{area}.csv",
    CHLA_TARGET: "track2_cha_train_point_{area}.csv",
}

TARGET_VALUE_COLUMNS = {
    TURBIDITY_TARGET: "turb_value",
    CHLA_TARGET: "cha_value",
}

PUBLIC_TEST_LABEL_STATS = {
    TURBIDITY_TARGET: {"count": 365, "min": 0.1, "median": 1.6, "mean": 2.1874, "max": 22.8},
    CHLA_TARGET: {"count": 103, "min": 0.18, "median": 1.42, "mean": 1.6159, "max": 5.3},
}
