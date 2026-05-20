"""Project paths and shared constants for the DO-warming analyses."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_START_KA = 0.0
ANALYSIS_END_KA = 640.0
BIN_WIDTH_KA = 0.2

STRONG_CSV = PROJECT_ROOT / "data/raw/rousseau_2023_ks_0p4_4kyr_strong_monsoon_start_times.csv"
WEAK_CSV = PROJECT_ROOT / "data/raw/rousseau_2023_ks_0p4_4kyr_weak_monsoon_start_times.csv"
LR04_XLSX = PROJECT_ROOT / "data/raw/lr04.xlsx"
CO2_XLSX = PROJECT_ROOT / "data/raw/composite_co2.xlsx"
PRE_TXT = PROJECT_ROOT / "data/raw/pre_1000_60_inter100.txt"

DATASET_SETTINGS = {
    "strong_monsoon_start": {
        "label": "Strong monsoon starts",
        "path": STRONG_CSV,
        "color": "#d95f02",
    },
    "weak_monsoon_start": {
        "label": "Weak monsoon starts",
        "path": WEAK_CSV,
        "color": "#6a3d9a",
    },
}
