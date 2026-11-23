"""
step5_evaluate.py
==================
PURPOSE : Evaluate the best trained model on the held-out TEST set.
          Plots:
            1. 24×24 Confusion Matrix (saved as results/confusion_matrix.png)
            2. Reliability / Calibration Diagram  (saved as results/reliability_diagram.png)
            3. Training curves (loss + accuracy)   (saved as results/training_curves.png)
RUN     : python scripts/step5_evaluate.py

The reliability diagram shows whether the model's confidence scores
actually match real accuracy.  A perfectly calibrated model would sit
on the diagonal.  Points above → underconfident; below → overconfident.
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works on all machines
import matplotlib.pyplot as plt

from psl_translator.config  import TEST_DIR, MODEL_PATH as BEST_MODEL, LABELS_JSON, TRAINING_LOG as LOG_PATH, RESULTS_DIR as Eval_DIR
from psl_translator.dataset import SignDataset, load_label_map
from psl_translator.model   import SignTCN

os.makedirs(Eval_DIR, exist_ok=True)
OUT_CM      = os.path.join(Eval_DIR, "confusion_matrix.png")
OUT_REL     = os.path.join(Eval_DIR, "reliability_diagram.png")
OUT_CURVES  = os.path.join(Eval_DIR, "training_curves.png")

# ── load label map ─────────────────────────────────────────────────────────────
label_map   = load_label_map(LABELS_JSON)
num_classes = len(label_map)
class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

# ── load model ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SignTCN(input_size=258, num_classes=num_classes)
model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
model.to(device)
model.eval()   # inference mode: no dropout, frozen BN stats

# ── test dataset ───────────────────────────────────────────────────────────────
test_dataset = SignDataset(TEST_DIR, label_map=label_map)
if len(test_dataset) == 0:
    raise SystemExit("No test data found — run steps 1-2 first (see data/README.md).")
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)
print(f"Test samples: {len(test_dataset)}")

# ── collect predictions ────────────────────────────────────────────────────────
all_true        = []    # true class indices
all_pred        = []    # predicted class indices
all_confidences = []    # max softmax probability (model's stated confidence)
all_correct     = []    # 1 if prediction was correct, else 0

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)                           # (batch, num_classes)
        probs  = torch.softmax(logits, dim=1)       # convert logits → probs

        confs, preds = probs.max(dim=1)             # top confidence + index

        all_true.extend(y.tolist())
        all_pred.extend(preds.cpu().tolist())
        all_confidences.extend(confs.cpu().tolist())
        all_correct.extend((preds.cpu() == y).tolist())

all_true        = np.array(all_true)
all_pred        = np.array(all_pred)
all_confidences = np.array(all_confidences)
all_correct     = np.array(all_correct)

# overall test accuracy
test_acc = 100.0 * all_correct.mean()
print(f"Test accuracy : {test_acc:.2f}%\n")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(true, pred, class_names, save_path):
    n  = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true, pred):
        cm[t][p] += 1

    # normalise rows to percentages for readability
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1   # avoid divide-by-zero
    cm_norm /= row_sums

    fig, ax = plt.subplots(figsize=(max(10, n), max(8, n - 2)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label",      fontsize=11)
    ax.set_title(f"Confusion Matrix (Test set)  —  Accuracy: {test_acc:.1f}%",
                 fontsize=13, fontweight="bold")

    # annotate each cell with the raw count
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


plot_confusion_matrix(all_true, all_pred, class_names, OUT_CM)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Reliability / Calibration Diagram
# ══════════════════════════════════════════════════════════════════════════════

def plot_reliability_diagram(confidences, correct, save_path, n_bins=10):
    """
    Bin predictions by confidence, plot mean confidence vs actual accuracy.
    A perfectly calibrated model lies on the diagonal (y = x).
    """
    bins     = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc  = []
    bin_conf = []
    bin_cnt  = []

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc.append(correct[mask].mean())
        bin_conf.append(confidences[mask].mean())
        bin_cnt.append(mask.sum())

    fig, ax = plt.subplots(figsize=(7, 6))

    # diagonal reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")

    # reliability bars
    ax.bar(
        [b - 0.05 for b in bin_conf],  # centre bars on mean confidence
        bin_acc,
        width=0.09,
        alpha=0.7,
        color="steelblue",
        label="Model"
    )

    # annotate bar with sample count
    for conf, acc, cnt in zip(bin_conf, bin_acc, bin_cnt):
        ax.text(conf - 0.05 + 0.045, acc + 0.01, f"n={cnt}",
                ha="center", fontsize=7, color="navy")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Mean confidence (max softmax probability)", fontsize=11)
    ax.set_ylabel("Fraction correct (actual accuracy)",        fontsize=11)
    ax.set_title("Reliability Diagram\n"
                 "(bars above diagonal → underconfident; "
                 "below → overconfident)",
                 fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Reliability diagram saved → {save_path}")


plot_reliability_diagram(all_confidences, all_correct, OUT_REL)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Training Curves (loss + accuracy over epochs)
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(log_path, save_path):
    if not os.path.exists(log_path):
        print("training_log.json not found — skipping curves plot.")
        return

    with open(log_path) as f:
        history = json.load(f)

    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    train_acc  = [h["train_acc"]  for h in history]
    val_acc    = [h["val_acc"]    for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # loss
    ax1.plot(epochs, train_loss, label="Train loss", color="royalblue")
    ax1.plot(epochs, val_loss,   label="Val loss",   color="tomato")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss over epochs"); ax1.legend()

    # accuracy
    ax2.plot(epochs, train_acc, label="Train acc", color="royalblue")
    ax2.plot(epochs, val_acc,   label="Val acc",   color="tomato")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy over epochs"); ax2.legend()

    plt.suptitle("Training History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved → {save_path}")


plot_training_curves(LOG_PATH, OUT_CURVES)

print("\nStep 5 complete. Check the results/ folder for all plots.")
print("Next → run step6_realtime.py")
