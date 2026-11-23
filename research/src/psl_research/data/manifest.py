"""
Build data/metadata/clips.csv — one row per raw video.

Signer and take come from, in order:
  1. data/metadata/signer_map.csv (path, signer, take), for files whose names
     carry no signer ID, e.g. WhatsApp exports. Made by
     scripts/dataset_tools/export_signer_map.py.
  2. the file name: data/raw/<sign>/<signer> (<take>).mp4, e.g. eat/3 (12).mp4
     (also accepted: s3_12.mp4, 3_12.mp4, 3-12.mp4)
Files matching neither (e.g. accidental "... - Copy.mp4" duplicates) are
skipped and listed.
"""

import csv
import re
from pathlib import Path

from .. import paths

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_PATTERNS = [re.compile(r"^(\d+)\s*\((\d+)\)"), re.compile(r"^s?(\d+)[_\- ](\d+)", re.I)]


def parse_name(stem: str):
    for pat in _PATTERNS:
        m = pat.match(stem)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def load_signer_map(path: Path) -> dict[str, tuple[int, int]]:
    if not path.exists():
        return {}
    with open(path) as fh:
        return {r["path"]: (int(r["signer"]), int(r["take"])) for r in csv.DictReader(fh)}


def build_manifest(raw_dir: Path = paths.RAW, out: Path = paths.METADATA / "clips.csv",
                   signer_map: Path = paths.METADATA / "signer_map.csv") -> list[dict]:
    mapping = load_signer_map(signer_map)
    rows, skipped = [], []
    for sign_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for f in sorted(sign_dir.iterdir()):
            if f.suffix.lower() not in VIDEO_EXT:
                continue
            rel = str(f.relative_to(raw_dir))
            parsed = mapping.get(rel) or parse_name(f.stem)
            if parsed is None:
                skipped.append(rel)
                continue
            signer, take = parsed
            rows.append({"clip_id": f"{sign_dir.name}/{f.stem}", "sign": sign_dir.name,
                         "signer": signer, "take": take, "path": rel})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip_id", "sign", "signer", "take", "path"])
        w.writeheader()
        w.writerows(rows)
    if skipped:
        print(f"Skipped {len(skipped)} file(s) with no signer: {skipped[:5]}")
    return rows


def load_manifest(path: Path = paths.METADATA / "clips.csv") -> list[dict]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["signer"], r["take"] = int(r["signer"]), int(r["take"])
    return rows
