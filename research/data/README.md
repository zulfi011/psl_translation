# Data

Nothing in `raw/`, `interim/`, `processed/` or `external/` is committed to git.

| Folder | Contents | Created by | Published? |
|---|---|---|---|
| `raw/` | main dataset: 24 signs × 50 clips + `noclass` (62), **never modified** | you | no: identifiable signers (see STRATEGY.md) |
| `raw_extras/` | additional recordings: extra takes, plus signs not in the main set (`drink`, `book_open`) | you | no |
| `samples/` | single test clips, incl. `hurais_sample/` (one clip per sign from another signer): use for out-of-sample / demo testing, never for training | you | no |
| `metadata/collection/` | original vocabulary list and recording strategy (private) | you | no |
| `metadata/` | `clips.csv` (clip, sign, anonymous signer number, take); `signer_map.csv` (signer number for files whose names carry none) | `build_manifest.py`, `dataset_tools/export_signer_map.py` | yes |
| `interim/<extractor>/` | full-length landmark arrays `(T, F)`, one folder per extractor | `extract_landmarks.py` | yes: the released dataset |
| `processed/<pipeline>/` | optional cache of model-ready arrays | — | no (re-creatable) |
| `splits/` | split definitions (JSON lists of clip IDs) | `make_splits.py` | yes |
| `external/` | other public datasets, if used for pre-training | you | follow their licence |

## Raw video naming

```
raw/
├── eat/
│   ├── 1 (1).mp4        ← signer 1, take 1
│   ├── 1 (2).mp4
│   └── 5 (10).mp4
├── go/
└── …                    ← one folder per sign (folder name = label)
```

`s1_01.mp4`, `1_01.mp4` and `1-01.mp4` are also accepted. Signer numbers must be
consistent across signs: signer 3 must be the same person in every folder.

**Current raw data** keeps its original WhatsApp file names, which carry no
signer ID. Signers were recovered from faces instead and stored in
`metadata/signer_map.csv` (anonymous numbers only), which `build_manifest.py`
reads first:

```bash
python scripts/dataset_tools/face_embed.py --clips <clip list> --models data/external/face_models --out <review>/faces.npz
python scripts/dataset_tools/assign_signers.py --faces <review>/faces.npz --clips <clip list> --out <review>/signers.csv
python scripts/dataset_tools/signer_sheets.py --faces <review>/faces.npz --signers <review>/signers.csv   # check by eye
python scripts/dataset_tools/export_signer_map.py --signers <review>/signers.csv
```

`<review>` = `metadata/collection/signer_review/` (private: holds face crops).
Face models: OpenCV YuNet + SFace (`opencv/face_detection_yunet`,
`opencv/face_recognition_sface` on Hugging Face) in `data/external/face_models/`.
