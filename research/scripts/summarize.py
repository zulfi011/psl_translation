"""Step 5 — aggregate fold metrics: mean ± std across folds, plus the leaderboard.

    python scripts/summarize.py exp002_tcn_v2          # one experiment
    python scripts/summarize.py                         # all experiments → reports/leaderboard.md
"""
import json
import sys

import numpy as np

from psl_research import paths

KEYS = ["accuracy", "macro_f1", "top5_accuracy", "ece"]


def load(exp):
    return [json.loads(p.read_text()) for p in sorted((paths.RUNS / exp).glob("fold*/metrics.json"))]


def row(exp, ms):
    cells = [f"{100*np.mean([m[k] for m in ms]):.1f} ± {100*np.std([m[k] for m in ms]):.1f}" for k in KEYS]
    return f"| {exp} | {len(ms)} | " + " | ".join(cells) + f" | {ms[0]['params']:,} |"


header = "| experiment | folds | accuracy % | macro-F1 % | top-5 % | ECE % | params |\n|---|---|---|---|---|---|---|"
exps = sys.argv[1:] or sorted(p.name for p in paths.RUNS.iterdir() if p.is_dir())
lines = [header] + [row(e, ms) for e in exps if (ms := load(e))]
print("\n".join(lines))
for e in exps:
    for m in load(e):
        print(f"  {e} fold{m['fold']} (test signer {m['test_signer']}): acc {100*m['accuracy']:.1f}%  F1 {100*m['macro_f1']:.1f}%")
if not sys.argv[1:]:
    paths.REPORTS.mkdir(exist_ok=True)
    (paths.REPORTS / "leaderboard.md").write_text("# Leaderboard\n\n" + "\n".join(lines) + "\n")
    print(f"→ {paths.REPORTS / 'leaderboard.md'}")
