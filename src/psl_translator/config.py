"""
config.py
=========
Central place for every path and setting used by the pipeline.

All paths are relative to the repository root, so the project works on any
machine without editing hard-coded Windows paths. Each path can be overridden
with an environment variable (or a ``.env`` file in the repo root).
"""

import os
from pathlib import Path

try:  # optional: load variables from a .env file if python-dotenv is installed
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def _repo_root() -> Path:
    env_root = os.getenv("PSL_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    # src/psl_translator/config.py -> repo root is two levels above "src"
    return Path(__file__).resolve().parents[2]


ROOT = _repo_root()

if load_dotenv is not None:
    load_dotenv(ROOT / ".env")


def _path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    return Path(value).expanduser().resolve() if value else default


# ── data ───────────────────────────────────────────────────────────────────────
DATA_DIR      = _path("PSL_DATA_DIR", ROOT / "data")
RAW_DIR       = _path("PSL_RAW_DIR", DATA_DIR / "raw")              # raw .mp4 clips, one folder per sign
LANDMARKS_DIR = _path("PSL_LANDMARKS_DIR", DATA_DIR / "landmarks")  # step 1 output
SPLIT_DIR     = _path("PSL_SPLIT_DIR", DATA_DIR / "split")          # step 2 output
TRAIN_DIR     = SPLIT_DIR / "train"
VAL_DIR       = SPLIT_DIR / "val"
TEST_DIR      = SPLIT_DIR / "test"

# ── model artefacts ────────────────────────────────────────────────────────────
MODELS_DIR   = _path("PSL_MODELS_DIR", ROOT / "models")
MODEL_PATH   = _path("PSL_MODEL_PATH", MODELS_DIR / "best_model.pth")
LABELS_JSON  = _path("PSL_LABELS_JSON", MODELS_DIR / "labels.json")
TRAINING_LOG = MODELS_DIR / "training_log.json"

# ── evaluation output ──────────────────────────────────────────────────────────
RESULTS_DIR = _path("PSL_RESULTS_DIR", ROOT / "results")

# ── LLM (OpenRouter) ───────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

# ── recognition / sliding-window settings ──────────────────────────────────────
SEQUENCE_LEN      = 60     # frames fed to the model
SIGN_WINDOW_SEC   = 3.0    # rolling window length in seconds
ADVANCE_SEC       = 1.0    # run inference every N seconds
CONFIDENCE_THRESH = 0.50   # minimum softmax confidence to accept a sign


def video_source(default: str | int = 0) -> str | int:
    """
    Video source for the real-time scripts: ``PSL_VIDEO_SOURCE`` env var,
    either a file path or a webcam index (e.g. ``0``).
    """
    value = os.getenv("PSL_VIDEO_SOURCE")
    if value is None or value == "":
        return default
    return int(value) if value.isdigit() else value
