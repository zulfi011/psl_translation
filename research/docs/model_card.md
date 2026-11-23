# Model card (draft)

**Status:** template, fill in from `reports/leaderboard.md` before release.

| | |
|---|---|
| Architecture | _e.g. TCN, 3 dilated residual blocks_ |
| Parameters | _e.g. 264,024_ |
| Input | 60 frames × 261 features (MediaPipe hands + pose, body-normalised, presence flags) |
| Output | 24 PSL sign classes |
| Training data | PSL landmarks dataset, signers _…_ |
| Evaluation | LOSO, 5 folds: accuracy _x ± y_ %, macro-F1 _x ± y_ % |
| CPU latency | _ms per window_ |

## Known limitations
- Vocabulary limited to 24 signs; out-of-vocabulary signs are forced into a known class or `noclass`.
- Trained on 5 signers; accuracy varies by signer (see per-fold results).
- Sensitive to lighting and occlusion through MediaPipe detection quality.
