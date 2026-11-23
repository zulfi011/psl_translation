"""Step 3 — write split definitions to data/splits/.

    python scripts/make_splits.py --strategy loso --val next_signer --name loso_nextval
    python scripts/make_splits.py --strategy holdout_v1 --name holdout_v1
"""
import argparse

from psl_research.data.manifest import load_manifest
from psl_research.data.splits import STRATEGIES, save_split

ap = argparse.ArgumentParser()
ap.add_argument("--strategy", choices=STRATEGIES, default="loso")
ap.add_argument("--name", required=True)
ap.add_argument("--val", default="next_signer", help="loso only: next_signer | fraction")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

kw = {"seed": args.seed}
if args.strategy == "loso":
    kw["val"] = args.val
split = STRATEGIES[args.strategy](load_manifest(), **kw)
path = save_split(split, args.name)
for f in split["folds"]:
    print(f"fold {f['fold']}: test signer {f['test_signer']}  train {len(f['train'])}  val {len(f['val'])}  test {len(f['test'])}")
print(f"→ {path}")
