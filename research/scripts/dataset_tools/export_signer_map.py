"""Turn the reviewed signer assignment into the public signer map.

Input:  signer_review/signers.csv (private; from assign_signers.py)
Output: data/metadata/signer_map.csv with columns path, signer, take
        - signer: anonymous number 1..N (no names, no gender)
        - take:   1..n per (sign, signer), in recording-time order

build_manifest.py reads this map for raw files whose names carry no signer
ID (the WhatsApp exports).

    python scripts/dataset_tools/export_signer_map.py \
        --signers data/metadata/collection/signer_review/signers.csv
"""
import argparse
import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_WA = re.compile(r"(\d{4}-\d{2}-\d{2}) at (\d{1,2}\.\d{2}\.\d{2} [AP]M)(?: \((\d+)\))?")


def recorded_at(name):
    m = _WA.search(name)
    if not m:
        return (datetime.max, 0, name)
    t = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %I.%M.%S %p")
    return (t, int(m.group(3) or 0), name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signers", required=True)
    ap.add_argument("--out", default=str(ROOT / "data/metadata/signer_map.csv"))
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.signers)))
    # stable anonymous numbering: women's batch first, then men, each by ID
    people = sorted({r["signer"] for r in rows},
                    key=lambda p: (p[0] != "W", int(p[1:]) if p[1:].isdigit() else 99, p))
    number = {p: n + 1 for n, p in enumerate(people)}

    by = defaultdict(list)
    for r in rows:
        by[(r["sign"], r["signer"])].append(r["file"])
    out = []
    for (sign, who), files in by.items():
        for take, f in enumerate(sorted(files, key=recorded_at), 1):
            out.append({"path": f"{sign}/{f}", "signer": number[who], "take": take})
    out.sort(key=lambda r: (r["path"].split("/")[0], r["signer"], r["take"]))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "signer", "take"])
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} clips -> {a.out}")
    print("signer numbers: " + ", ".join(f"{p}={n}" for p, n in number.items()))


if __name__ == "__main__":
    main()
