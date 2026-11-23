"""Step 4 — train an experiment (all folds by default).

    python scripts/train.py configs/experiments/exp001_tcn_v1_baseline.yaml
    python scripts/train.py configs/experiments/exp002_tcn_v2.yaml --fold 0
    python scripts/train.py configs/experiments/exp002_tcn_v2.yaml --set train.lr=0.0005 --name exp002b
"""
import argparse

import yaml

from psl_research.config import load_experiment
from psl_research.data.splits import load_split
from psl_research.training.trainer import train_fold

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--fold", type=int, default=None)
ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. model.params.dropout=0.2")
ap.add_argument("--name", default=None, help="run name (defaults to the config's name)")
args = ap.parse_args()

overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
cfg = load_experiment(args.config, overrides)
if args.name:
    cfg["name"] = args.name

n_folds = len(load_split(cfg["data"]["split"])["folds"])
for fold in ([args.fold] if args.fold is not None else range(n_folds)):
    train_fold(cfg, fold)
print(f"\nSummarise with:  python scripts/summarize.py {cfg['name']}")
