"""Review pictures for the signer assignment: one per batch (men, women).

Each row is one person, each column one sign; the tile is that person's
face in that sign (the clip closest to their mean face). Empty = the person
is not in that sign. Reads faces.npz (face_embed.py) and signers.csv
(assign_signers.py); writes check_men.jpg and check_women.jpg next to
signers.csv.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", required=True)
    ap.add_argument("--signers", required=True)
    ap.add_argument("--tile", type=int, default=96)
    a = ap.parse_args()

    F = np.load(a.faces)
    rows = list(csv.DictReader(open(a.signers)))
    out = Path(a.signers).parent
    t, lw, hh = a.tile, 150, 70
    signs = sorted({r["sign"] for r in rows}, key=str.lower)
    for gender, name in [("M", "men"), ("F", "women")]:
        people = sorted({r["signer"] for r in rows if r["gender"] == gender})
        count = Counter(r["signer"] for r in rows)
        W, H = lw + t * len(signs), hh + t * len(people)
        img = np.full((H, W, 3), 255, np.uint8)
        for c, s in enumerate(signs):  # vertical sign names
            lab = np.full((24, hh), 255, np.uint8)
            cv2.putText(lab, s, (2, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
            lab = cv2.rotate(lab, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x = lw + c * t + (t - 24) // 2
            img[:hh, x:x + 24] = cv2.cvtColor(lab, cv2.COLOR_GRAY2BGR)
        for r_, p in enumerate(people):
            y = hh + r_ * t
            cv2.putText(img, p, (10, y + t // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 0), 2)
            cv2.putText(img, f"{count[p]} clips", (10, y + t // 2 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
            for c, s in enumerate(signs):
                mine = [i for i, r in enumerate(rows)
                        if r["signer"] == p and r["sign"] == s]
                if not mine:
                    continue
                i = max(mine, key=lambda i: float(rows[i]["sim"]))
                img[y:y + t, lw + c * t:lw + (c + 1) * t] = cv2.resize(
                    F["crop"][i], (t, t))
            cv2.line(img, (0, y), (W, y), (200, 200, 200), 1)
        cv2.imwrite(str(out / f"check_{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"check_{name}.jpg: {len(people)} people")


if __name__ == "__main__":
    main()
