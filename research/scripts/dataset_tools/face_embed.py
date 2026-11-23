"""Compute one face embedding per clip, to recover signer identities.

Uses OpenCV's YuNet face detector and SFace face recogniser. Samples a few
frames per clip, keeps the largest face in each, and averages the
L2-normalised 128-d features.

Output (an .npz, kept out of git because it holds face crops):
    sign, clip   clip keys (sign folder, file name)
    emb          (N, 128) mean embedding per clip, L2-normalised
    nfaces       frames where a face was found
    crop         (N, 112, 112, 3) aligned face from the middle frame

Usage:
    python scripts/dataset_tools/face_embed.py \
        --clips data/metadata/collection/signer_review/signers_guess.csv \
        --models data/external/face_models \
        --out data/metadata/collection/signer_review/faces.npz
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]


def largest_face(det, img):
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None
    return faces[np.argmax(faces[:, 2] * faces[:, 3])]


def embed_clip(path, det, rec, n_frames):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    feats, crop = [], None
    idx = np.linspace(0.1 * total, 0.9 * total, n_frames).astype(int)
    for k, i in enumerate(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if not ok:
            continue
        face = largest_face(det, img)
        if face is None:
            continue
        aligned = rec.alignCrop(img, face)
        f = rec.feature(aligned).ravel()
        feats.append(f / np.linalg.norm(f))
        if crop is None or k == n_frames // 2:
            crop = aligned
    cap.release()
    if not feats:
        return np.zeros(128, np.float32), 0, np.zeros((112, 112, 3), np.uint8)
    m = np.mean(feats, axis=0)
    return (m / np.linalg.norm(m)).astype(np.float32), len(feats), crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="csv with sign,file columns")
    ap.add_argument("--raw", default=str(ROOT / "data/raw"))
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=6)
    a = ap.parse_args()

    models = Path(a.models)
    det = cv2.FaceDetectorYN.create(
        str(models / "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.6
    )
    rec = cv2.FaceRecognizerSF.create(
        str(models / "face_recognition_sface_2021dec.onnx"), ""
    )

    rows = list(csv.DictReader(open(a.clips)))
    embs, nf, crops = [], [], []
    for r in tqdm(rows, desc="faces"):
        e, n, c = embed_clip(Path(a.raw) / r["sign"] / r["file"], det, rec, a.frames)
        embs.append(e), nf.append(n), crops.append(c)

    np.savez_compressed(
        a.out,
        sign=np.array([r["sign"] for r in rows]),
        clip=np.array([r["file"] for r in rows]),
        emb=np.stack(embs),
        nfaces=np.array(nf),
        crop=np.stack(crops),
    )
    print(f"{len(rows)} clips, no face found in {sum(n == 0 for n in nf)}")


if __name__ == "__main__":
    main()
