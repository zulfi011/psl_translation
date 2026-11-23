"""
Recover who signed each clip when file names don't say (WhatsApp exports).

Each sign was recorded in recording batches (one batch per day). Within one
batch every person wears the same clothes in the same room, so clips are
grouped by colour appearance of a few frames. The number of people per batch
is clips / clips_per_signer.

Output (all git-ignored, contain faces):
  data/metadata/collection/signer_review/clusters.csv   clip → sign, batch, cluster
  data/metadata/collection/signer_review/<sign>.jpg     contact sheet, coloured by cluster
  data/metadata/collection/signer_review/people.jpg     one face crop per cluster per sign,
                                                       used to give clusters global signer IDs

    python scripts/dataset_tools/cluster_signers.py --clips-per-signer 10
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from psl_research import paths

PAT = re.compile(r"(\d{4}-\d\d-\d\d) at (\d+)\.(\d+)\.(\d+) (AM|PM)(?: \((\d+)\))?")
COLORS = [(0, 200, 255), (255, 120, 0), (0, 220, 0), (220, 0, 220), (0, 0, 255), (255, 255, 0)]


def send_key(name):
    m = PAT.search(name)
    d, h, mi, s, ap, n = m.groups()
    return d, int(h) % 12 + (12 if ap == "PM" else 0), int(mi), int(s), int(n or 0)


def frames(path, positions=(0.15, 0.5, 0.85)):
    cap = cv2.VideoCapture(str(path))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    out = []
    for p in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * p))
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out


def appearance(fs):
    """Colour histogram of the central body strip (clothes) + full frame (room)."""
    feats = []
    for f in fs:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        body = hsv[int(h * .3):int(h * .9), int(w * .25):int(w * .75)]
        for region in (body, hsv):
            hist = cv2.calcHist([region], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256]).ravel()
            feats.append(hist / (hist.sum() + 1e-9))
    return np.sqrt(np.mean(np.stack(feats).reshape(len(fs), -1), axis=0))   # Hellinger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(paths.RAW))
    ap.add_argument("--clips-per-signer", type=int, default=10)
    ap.add_argument("--out", default=str(paths.METADATA / "collection" / "signer_review"))
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, faces = [], defaultdict(list)
    for sign_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        files = sorted((f for f in sign_dir.iterdir() if f.suffix == ".mp4" and "- Copy" not in f.name),
                       key=lambda f: send_key(f.name))
        batches = defaultdict(list)
        for f in files:
            batches[send_key(f.name)[0]].append(f)
        tiles = []
        for b_idx, (day, bfiles) in enumerate(sorted(batches.items())):
            k = max(1, round(len(bfiles) / args.clips_per_signer))
            fr = [frames(f) for f in bfiles]
            X = np.stack([appearance(x) for x in fr])
            labels = (AgglomerativeClustering(n_clusters=k, linkage="average", metric="cosine").fit_predict(X)
                      if k > 1 else np.zeros(len(bfiles), int))
            # renumber clusters by first appearance in send order → stable IDs
            order = {c: i for i, c in enumerate(dict.fromkeys(labels))}
            labels = [order[c] for c in labels]
            for f, lab, x in zip(bfiles, labels, fr):
                cid = f"{day}_c{lab}"
                rows.append({"sign": sign_dir.name, "file": f.name, "batch": day, "cluster": cid})
                mid = x[len(x) // 2]
                tile = cv2.resize(mid, (90, 160))
                cv2.rectangle(tile, (0, 0), (89, 159), COLORS[lab % len(COLORS)], 6)
                tiles.append(tile)
                if len(faces[(sign_dir.name, cid)]) == 0:
                    h, w = mid.shape[:2]
                    faces[(sign_dir.name, cid)].append(cv2.resize(mid[: int(h * .45), int(w * .15): int(w * .85)], (110, 106)))
        cols = 10
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / f"{sign_dir.name}.jpg"),
                    np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))
        sizes = {c: sum(r["cluster"] == c for r in rows if r["sign"] == sign_dir.name)
                 for c in dict.fromkeys(r["cluster"] for r in rows if r["sign"] == sign_dir.name)}
        print(f"{sign_dir.name:9s} " + "  ".join(f"{c}:{n}" for c, n in sizes.items()))

    with open(out / "clusters.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sign", "file", "batch", "cluster"])
        w.writeheader()
        w.writerows(rows)

    # people sheet: one row per sign, one face per cluster (batch order)
    sheet = []
    for sign in sorted({s for s, _ in faces}):
        cids = sorted(c for s, c in faces if s == sign)
        row = [faces[(sign, c)][0] for c in cids]
        while len(row) < 7:
            row.append(np.zeros_like(row[0]))
        label = np.zeros((106, 110, 3), np.uint8)
        cv2.putText(label, sign, (4, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        sheet.append(np.hstack([label] + row[:7]))
    cv2.imwrite(str(out / "people.jpg"), np.vstack(sheet))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
