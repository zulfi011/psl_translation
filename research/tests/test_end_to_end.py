"""Smoke test: synthetic landmarks → splits → train every model → metrics."""
import csv

import numpy as np
import pytest

from psl_research import paths
from psl_research.config import load_experiment
from psl_research.data.splits import loso, save_split
from psl_research.data.manifest import load_manifest
from psl_research.training.trainer import train_fold

SIGNS, SIGNERS, TAKES = ["eat", "go", "salam"], [1, 2, 3], 4


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.default_rng(0)
    rows = []
    for si, sign in enumerate(SIGNS):
        for signer in SIGNERS:
            for take in range(1, TAKES + 1):
                cid = f"{sign}/{signer} ({take})"
                x = rng.uniform(0.3, 0.7, (int(rng.integers(40, 90)), 258)).astype(np.float32)
                x[:, si * 10:(si + 1) * 10] += 0.2            # class signal
                out = paths.INTERIM / "mediapipe_holistic_v1" / f"{cid}.npy"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, x)
                rows.append({"clip_id": cid, "sign": sign, "signer": signer, "take": take, "path": f"{cid}.mp4"})
    paths.METADATA.mkdir(parents=True, exist_ok=True)
    with open(paths.METADATA / "clips.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    save_split(loso(load_manifest()), "loso_nextval")


@pytest.mark.parametrize("exp", ["exp001_tcn_v1_baseline.yaml", "exp002_tcn_v2.yaml", "exp003_tcn_v1_v2data.yaml"])
def test_experiment_runs(synthetic_data, exp):
    import shutil
    shutil.copytree(_real_configs(), paths.CONFIGS, dirs_exist_ok=True)
    cfg = load_experiment(f"experiments/{exp}", {"train.epochs": 2, "train.device": "cpu"})
    m = train_fold(cfg, fold=0)
    assert 0 <= m["accuracy"] <= 1 and m["n"] == len(SIGNS) * TAKES and m["params"] > 0


def _real_configs():
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "configs"
