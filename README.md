# PSL — Pakistani Sign Language Translator

Real-time recognition of **Pakistani Sign Language (PSL)** from video, turning
recognised signs into natural English sentences.

Built by **Abdul Rehman**.

```
video ──► MediaPipe Holistic ──► hand + pose landmarks ──► TCN classifier ──► sign tokens ──► LLM ──► English sentences
```

## Highlights

- **24 everyday PSL signs** (I, come, eat, friend, go, good, help, home, learn,
  like, morning, no, phone, please, salam, school, sorry, student, teacher,
  thank you, want, water, you, + idle) recorded from multiple signers.
- **Lightweight model**: a Temporal Convolutional Network on MediaPipe
  landmarks (≈0.5 M parameters), fast enough for real-time use on a CPU.
- **End to end**: live webcam recognition with a sliding window, a Flask API,
  and an LLM step that turns sign sequences into fluent sentences.
- **Research-grade v2 in progress**: signer-independent evaluation and a
  redesigned training pipeline.

## Repository layout

| Folder | What it is |
|---|---|
| [`fyp/`](fyp/) | **v1, the final-year-project system**: the trained TCN model, the real-time demo, the Flask API and the LLM sentence step. Frozen as the baseline. |
| [`research/`](research/) | **v2, the research codebase**: versioned data processing, pluggable models and preprocessing, leave-one-signer-out evaluation, and experiment tracking. |

## Results so far (v1)

| Metric | Value |
|---|---|
| Classes | 24 |
| Train accuracy | 99.5 % |
| Validation accuracy (held-out signer group) | **84.2 %** |
| Parameters | 497,240 |

Details and training curves: [`fyp/README.md`](fyp/README.md).

## Status: retraining (v2)

v1 works, but a review of it turned up issues worth fixing before publishing
final numbers:

- the data augmentation corrupted missing-hand frames and did not swap left
  and right hands when mirroring;
- training and live inference preprocessed landmarks differently;
- accuracy was measured on a single held-out group of clips.

v2 ([`research/`](research/)) fixes the pipeline, normalises landmarks
relative to the body, and reports accuracy as **mean ± std over
leave-one-signer-out folds**, testing every model on a person it never saw
during training. Results, trained weights and the landmark dataset will be
released here when the experiments finish.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e fyp                 # v1 system
pip install -e "research[dev]"     # v2 research code
```

- Run the v1 demo / API: see [`fyp/README.md`](fyp/README.md)
- Run the v2 pipeline: see [`research/README.md`](research/README.md)

## License

Code: [MIT](LICENSE) © 2026 Abdul Rehman
