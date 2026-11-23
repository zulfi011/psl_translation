"""
Experiment config loading.

An experiment YAML composes reusable pieces by file reference and may override
any field:

    name: exp001_tcn_baseline
    data:  data/landmarks_v2.yaml      # path relative to configs/
    model: model/tcn.yaml
    train: train/default.yaml
    overrides:
      model.params.dropout: 0.3
"""

import copy
import hashlib
import json
from pathlib import Path

import yaml

from . import paths


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve(value):
    if isinstance(value, str) and value.endswith((".yaml", ".yml")):
        return _load_yaml(paths.CONFIGS / value)
    return copy.deepcopy(value)


def _set_dotted(cfg: dict, dotted: str, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_experiment(path: str | Path, extra_overrides: dict | None = None) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = paths.CONFIGS / path
    raw = _load_yaml(path)
    cfg = {k: _resolve(v) for k, v in raw.items() if k != "overrides"}
    cfg.setdefault("name", path.stem)
    for dotted, value in {**raw.get("overrides", {}), **(extra_overrides or {})}.items():
        _set_dotted(cfg, dotted, value)
    return cfg


def config_hash(cfg: dict, length: int = 8) -> str:
    """Stable short hash, used to version derived data and runs."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:length]


def save_config(cfg: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
