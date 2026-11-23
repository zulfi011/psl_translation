"""Assign every clip to a signer from face embeddings (see face_embed.py).

Two steps, run separately for the men's and the women's recording batches:

  1. Within each sign, split the clips by face into round(n / 10) people
     (each person recorded ~10 takes per sign). Inside one sign the people
     are easy to tell apart.
  2. Link those per-sign groups across signs: average-linkage clustering of
     the group mean embeddings, cut at a cosine-distance threshold. The
     number of people is not forced; it comes out of the data.

Writes signers.csv (sign, file, gender, signer, sign_group, sim) where sim is
the clip's cosine similarity to its person's mean face, and prints a per-sign
table plus the merge distances, so the gap between "same person" and
"different person" can be checked.

Optional overrides (from a human check) are applied last:
    --merge M5=M3            relabel all of M5 as M3
    --split teacher.0=M6     give one per-sign group its own ID
"""
import argparse
import csv
from collections import Counter, defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering


def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def per_sign_groups(emb, signs, idx, per_person):
    groups = []
    for s in sorted(set(signs[idx])):
        sub = idx[signs[idx] == s]
        k = max(1, round(len(sub) / per_person))
        lab = (np.zeros(len(sub), int) if k == 1 else
               AgglomerativeClustering(n_clusters=k, metric="cosine",
                                       linkage="average").fit_predict(emb[sub]))
        for j in range(k):
            groups.append((f"{s}.{j}", sub[lab == j]))
    return groups


def link(emb, groups, threshold):
    G = norm(np.stack([emb[i].mean(0) for _, i in groups]))
    D = np.clip(1 - G @ G.T, 0, 2)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), "average")
    return fcluster(Z, threshold, "distance"), Z[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", required=True)
    ap.add_argument("--clips", required=True,
                    help="csv with sign, file and person_guess (W = women's batch)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.35,
                    help="cosine distance above which groups are different people")
    ap.add_argument("--per-person", type=int, default=10)
    ap.add_argument("--outlier", type=float, default=0.5,
                    help="clips less similar than this to their person get moved")
    ap.add_argument("--merge", nargs="*", default=[])
    ap.add_argument("--split", nargs="*", default=[])
    a = ap.parse_args()

    F = np.load(a.faces)
    rows = list(csv.DictReader(open(a.clips)))
    assert [r["file"] for r in rows] == list(F["clip"])
    emb, signs = F["emb"], F["sign"]
    women = np.array([r["person_guess"] == "W" for r in rows])

    signer = np.empty(len(rows), object)
    group_of = np.empty(len(rows), object)
    for mask, prefix in [(women, "W"), (~women, "M")]:
        idx = np.where(mask)[0]
        groups = per_sign_groups(emb, signs, idx, a.per_person)
        lab, heights = link(emb, groups, a.threshold)
        top = np.sort(heights)[::-1][:8]
        print(f"{prefix}: {len(groups)} per-sign groups -> {lab.max()} people; "
              f"largest merge distances {np.round(top, 2).tolist()}")
        sizes = Counter()
        for (_, i), p in zip(groups, lab):
            sizes[p] += len(i)
        name = {p: f"{prefix}{n + 1}" for n, (p, _) in enumerate(sizes.most_common())}
        for (gname, i), p in zip(groups, lab):
            signer[i] = name[p]
            group_of[i] = gname

    for m in a.merge:
        src, dst = m.split("=")
        signer[signer == src] = dst
    for s in a.split:
        g, new = s.split("=")
        signer[group_of == g] = new

    # A single odd clip can be forced into a per-sign group it does not
    # belong to (e.g. one take by a third person). Move clips that match
    # their own person poorly to the best-matching person of the same batch.
    protos = {p: norm(emb[signer == p].mean(0)) for p in set(signer)}
    for i in range(len(rows)):
        if emb[i] @ protos[signer[i]] < a.outlier:
            same = [p for p in protos if p[0] == signer[i][0]]
            best = max(same, key=lambda p: emb[i] @ protos[p])
            print(f"moved {rows[i]['sign']}/{rows[i]['file']}: "
                  f"{signer[i]} -> {best}")
            signer[i] = best
    protos = {p: norm(emb[signer == p].mean(0)) for p in set(signer)}
    sim = np.array([emb[i] @ protos[signer[i]] for i in range(len(rows))])

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sign", "file", "gender", "signer", "sign_group", "sim"])
        for r, s, g, si, wo in zip(rows, signer, group_of, sim, women):
            w.writerow([r["sign"], r["file"], "F" if wo else "M", s, g, f"{si:.3f}"])

    table = defaultdict(Counter)
    for r, s in zip(rows, signer):
        table[r["sign"]][s] += 1
    people = sorted(set(signer))
    print("sign      " + " ".join(f"{p:>4}" for p in people))
    for s in sorted(table):
        print(f"{s:<10}" + " ".join(f"{table[s][p] or '.':>4}" for p in people))
    print("total     " + " ".join(f"{sum(t[p] for t in table.values()):>4}"
                                  for p in people))


if __name__ == "__main__":
    main()
