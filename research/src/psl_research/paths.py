"""Standard locations inside research/. Override the root with PSL_RESEARCH_ROOT."""

import os
from pathlib import Path

ROOT = Path(os.getenv("PSL_RESEARCH_ROOT", Path(__file__).resolve().parents[2])).resolve()

CONFIGS = ROOT / "configs"

DATA = ROOT / "data"
RAW = DATA / "raw"              # original videos — never modified
INTERIM = DATA / "interim"      # landmarks, one folder per extractor version
PROCESSED = DATA / "processed"  # optional cached, model-ready arrays per pipeline
SPLITS = DATA / "splits"        # split definitions (small JSON, can be published)
METADATA = DATA / "metadata"    # clips.csv manifest, signers.csv

RUNS = ROOT / "experiments" / "runs"
REPORTS = ROOT / "reports"
