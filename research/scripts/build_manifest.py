"""Step 1 — scan data/raw/ and write data/metadata/clips.csv.

    python scripts/build_manifest.py
"""
from collections import Counter

from psl_research import paths
from psl_research.data.manifest import build_manifest

rows = build_manifest()
print(f"{len(rows)} clips, {len({r['sign'] for r in rows})} signs, signers {sorted({r['signer'] for r in rows})}")
for (sign, n) in sorted(Counter(r["sign"] for r in rows).items()):
    print(f"  {sign:12s} {n}")
print(f"→ {paths.METADATA / 'clips.csv'}")
