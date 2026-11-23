"""
step4_train.py
===============
PURPOSE : Train the SignTCN model with early stopping, validation accuracy
          tracking, ReduceLROnPlateau scheduler, and label smoothing.
RUN     : python scripts/step4_train.py

Saves
-----
  models/best_model.pth   ← weights of the best epoch (lowest val loss)
  models/training_log.json ← epoch-by-epoch loss / accuracy history
  models/labels.json       ← already written by step2; re-used here

Early stopping monitors VALIDATION LOSS.
If it does not improve for PATIENCE=10 consecutive epochs, training stops
and the best weights are restored automatically.
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from psl_translator.config  import TRAIN_DIR, VAL_DIR, MODELS_DIR, MODEL_PATH as BEST_MODEL, TRAINING_LOG as LOG_PATH, LABELS_JSON
from psl_translator.dataset import SignDataset, load_label_map
from psl_translator.model   import SignTCN

os.makedirs(MODELS_DIR, exist_ok=True)

# ── hyper-parameters ───────────────────────────────────────────────────────────
BATCH_SIZE     = 8
LEARNING_RATE  = 0.001
MAX_EPOCHS     = 200     # upper ceiling — early stopping will halt before this
PATIENCE       = 10      # early stopping patience (epochs without improvement)
LABEL_SMOOTHING = 0.1   # prevents overconfidence on small dataset
DROPOUT        = 0.4

# ── reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)

# ── label map ──────────────────────────────────────────────────────────────────
label_map   = load_label_map(LABELS_JSON)
num_classes = len(label_map)
print(f"Classes: {label_map}")
print(f"Num classes: {num_classes}\n")

# ── datasets & loaders ────────────────────────────────────────────────────────
train_dataset = SignDataset(TRAIN_DIR, label_map=label_map)
val_dataset   = SignDataset(VAL_DIR,   label_map=label_map)

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise SystemExit("No training/validation data found — run steps 1-3 first (see data/README.md).")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train samples : {len(train_dataset)}")
print(f"Val   samples : {len(val_dataset)}\n")

# ── device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ── model ──────────────────────────────────────────────────────────────────────
model = SignTCN(input_size=258, num_classes=num_classes, dropout=DROPOUT)
model = model.to(device)

# ── loss, optimiser, scheduler ────────────────────────────────────────────────
# label_smoothing=0.1 prevents the model from becoming overconfident
loss_fn   = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# Adam with initial lr=0.001
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ReduceLROnPlateau: halve lr when val_loss has not improved for 5 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",        # we are minimising validation loss
    factor=0.5,        # multiply lr by 0.5
    patience=5,        # wait 5 epochs before reducing
)


# ── helper: one epoch of training ─────────────────────────────────────────────
def run_epoch(loader, is_train):
    """
    Run one full pass through the loader.
    Returns (average_loss, accuracy_percent).
    """
    if is_train:
        model.train()   # activates dropout + batchnorm training mode
    else:
        model.eval()    # disables dropout + uses running batchnorm stats

    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)                   # forward pass
            loss   = loss_fn(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)  # accumulate sum (not mean)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y).sum().item()
            total      += len(y)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ── training loop with early stopping ─────────────────────────────────────────
best_val_loss    = float("inf")
epochs_no_improve = 0
history          = []

print("=" * 60)
print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
      f"{'Val Loss':>9}  {'Val Acc':>8}  {'LR':>10}")
print("=" * 60)

training_start_time = time.time()
best_stats = {}

for epoch in range(1, MAX_EPOCHS + 1):

    epoch_start = time.time()

    train_loss, train_acc = run_epoch(train_loader, is_train=True)
    val_loss,   val_acc   = run_epoch(val_loader,   is_train=False)

    epoch_time = time.time() - epoch_start

    # step the scheduler based on validation loss
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>8.1f}%  "
          f"{val_loss:>9.4f}  {val_acc:>7.1f}%  {current_lr:>10.6f}")

    # save epoch stats for later plotting
    history.append({
        "epoch":      epoch,
        "train_loss": train_loss,
        "train_acc":  train_acc,
        "val_loss":   val_loss,
        "val_acc":    val_acc,
        "lr":         current_lr,
        "epoch_time": epoch_time,
    })

    # ── early stopping logic ──────────────────────────────────────────────────
    if val_loss < best_val_loss - 1e-4:   # must improve by at least 0.0001
        best_val_loss      = val_loss
        epochs_no_improve  = 0

        best_stats = {
        "epoch":      epoch,
        "train_loss": train_loss,
        "train_acc":  train_acc,
        "val_loss":   val_loss,
        "val_acc":    val_acc,
        }

        # save best weights
        torch.save(model.state_dict(), BEST_MODEL)
        print(f"         ↳ New best val_loss={best_val_loss:.4f}  — model saved.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs "
                  f"(no improvement for {PATIENCE} epochs).")
            break

# ── save training history ──────────────────────────────────────────────────────
with open(LOG_PATH, "w") as f:
    json.dump(history, f, indent=2)

total_time   = time.time() - training_start_time
epochs_done  = len(history)
avg_per_epoch = total_time / epochs_done if epochs_done else 0

print(f"\nTraining complete.")
print(f"  Epochs completed  : {epochs_done} out of {MAX_EPOCHS}")
print(f"  ── Best Epoch ({best_stats['epoch']}) ──")
print(f"  Train loss        : {best_stats['train_loss']:.4f}")
print(f"  Train acc         : {best_stats['train_acc']:.1f}%")
print(f"  Val loss          : {best_stats['val_loss']:.4f}")
print(f"  Val acc           : {best_stats['val_acc']:.1f}%")
print(f"  Total time        : {total_time:.1f}s  ({total_time/60:.1f} min)")
print(f"  Avg time / epoch  : {avg_per_epoch:.1f}s")
print(f"  Model saved       : {BEST_MODEL}")
print(f"  Log saved         : {LOG_PATH}")
print("\nNext → run step5_evaluate.py")
