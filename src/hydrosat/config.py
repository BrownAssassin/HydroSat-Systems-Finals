"""Shared configuration constants for the HydroSat final-round scaffold."""

from __future__ import annotations

from pathlib import Path

DEFAULT_DATA_ROOT = Path("data/raw")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_INPUT_DIR = Path("/input")
DEFAULT_OUTPUT_DIR = Path("/output")

TRAIN_ROOT_NAME = "train"
SAMPLE_INPUT_ROOT_NAME = "sample_input"
GUIDES_ROOT_NAME = "guides"

TRAIN_AREAS = ("area1", "area2", "area3", "area5", "area6", "area7")

TURBIDITY_TARGET = "turbidity"
CHLA_TARGET = "chla"

TURBIDITY_RESULT_FILENAME = "result_turbidity.json"
CHLA_RESULT_FILENAME = "result_chla.json"
REQUIRED_OUTPUT_FILENAMES = (TURBIDITY_RESULT_FILENAME, CHLA_RESULT_FILENAME)

TARGET_RESULT_FILENAMES = {
    TURBIDITY_TARGET: TURBIDITY_RESULT_FILENAME,
    CHLA_TARGET: CHLA_RESULT_FILENAME,
}

TARGET_POINT_FILENAMES = {
    TURBIDITY_TARGET: "track2_turb_test_point.csv",
    CHLA_TARGET: "track2_cha_test_point.csv",
}
