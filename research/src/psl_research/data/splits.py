"""
Signer-independent splits, written to data/splits/<name>.json.

  loso        leave-one-signer-out: fold k tests on signer k. Validation is either
              the next signer (val="next_signer", strictest) or a stratified
              fraction of the training signers' clips (val="fraction").
  holdout_v1  the FYP v1 protocol: train on signers 1-4, signer 5 split 50/50
              into val/test. Kept only for comparing against the v1 baseline.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from .. import paths


def _ids(rows):
    return sorted(r["clip_id"] for r in rows)


def _stratified_fraction(rows, frac, rng):
    by_sign = defaultdict(list)
    for r in rows:
        by_sign[r["sign"]].append(r)
    picked = []
    for sign in sorted(by_sign):
        group = sorted(by_sign[sign], key=lambda r: r["clip_id"])
        rng.shuffle(group)
        picked += group[:max(1, round(len(group) * frac))]
    return picked


def loso(manifest, val="next_signer", val_fraction=0.15, seed=42) -> dict:
    signers = sorted({r["signer"] for r in manifest})
    folds = []
    for i, test_s in enumerate(signers):
        rng = random.Random(seed + i)
        test = [r for r in manifest if r["signer"] == test_s]
        rest = [r for r in manifest if r["signer"] != test_s]
        if val == "next_signer":
            val_s = signers[(i + 1) % len(signers)]
            val_rows = [r for r in rest if r["signer"] == val_s]
        elif val == "fraction":
            val_rows = _stratified_fraction(rest, val_fraction, rng)
        else:
            raise ValueError(f"unknown val strategy {val!r}")
        val_set = {r["clip_id"] for r in val_rows}
        train = [r for r in rest if r["clip_id"] not in val_set]
        folds.append({"fold": i, "test_signer": test_s, "train": _ids(train), "val": _ids(val_rows), "test": _ids(test)})
    return {"strategy": "loso", "val": val, "seed": seed, "signers": signers, "folds": folds}


def holdout_v1(manifest, train_signers=(1, 2, 3, 4), holdout_signer=5, seed=42) -> dict:
    rng = random.Random(seed)
    train = [r for r in manifest if r["signer"] in train_signers]
    hold = [r for r in manifest if r["signer"] == holdout_signer]
    val, test = [], []
    by_sign = defaultdict(list)
    for r in hold:
        by_sign[r["sign"]].append(r)
    for sign in sorted(by_sign):
        g = sorted(by_sign[sign], key=lambda r: r["clip_id"])
        rng.shuffle(g)
        val += g[: len(g) // 2]
        test += g[len(g) // 2:]
    return {"strategy": "holdout_v1", "seed": seed,
            "folds": [{"fold": 0, "test_signer": holdout_signer, "train": _ids(train), "val": _ids(val), "test": _ids(test)}]}


STRATEGIES = {"loso": loso, "holdout_v1": holdout_v1}


def save_split(split: dict, name: str, out_dir: Path = paths.SPLITS) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(split, indent=1))
    return path


def load_split(name: str, split_dir: Path = paths.SPLITS) -> dict:
    return json.loads((split_dir / f"{name}.json").read_text())
