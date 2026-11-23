"""
Train + evaluate one fold of one experiment.

Writes experiments/runs/<experiment>/fold<k>/:
  config.yaml   resolved config (exact reproduction)
  labels.json   class → index
  history.json  per-epoch train/val loss & accuracy
  best.pt       best weights (by val loss)
  metrics.json  test metrics + model size
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .. import models  # noqa: F401  (register models)
from .. import paths
from ..config import save_config
from ..data.dataset import LandmarkDataset
from ..data.manifest import load_manifest
from ..data.splits import load_split
from ..evaluation.metrics import summarize
from ..registry import MODELS
from ..utils.seed import seed_everything


def _device(pref: str):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_epoch(model, loader, loss_fn, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum, probs, ys = 0, 0, 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
            if not train:
                probs.append(torch.softmax(logits, 1).cpu().numpy())
                ys.append(y.cpu().numpy())
    out = {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1)}
    if not train:
        out["probs"], out["y"] = np.concatenate(probs), np.concatenate(ys)
    return out


def train_fold(cfg: dict, fold: int, run_root: Path | None = None) -> dict:
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["train"]
    seed = train_cfg.get("seed", 42) + fold
    seed_everything(seed)

    manifest = load_manifest(paths.METADATA / data_cfg.get("manifest", "clips.csv"))
    classes = sorted({r["sign"] for r in manifest})
    label_map = {c: i for i, c in enumerate(classes)}
    split = load_split(data_cfg["split"])["folds"][fold]

    def ds(ids, train):
        return LandmarkDataset(ids, label_map, data_cfg["extractor"], data_cfg["transforms"], train=train, seed=seed)
    train_ds, val_ds, test_ds = ds(split["train"], True), ds(split["val"], False), ds(split["test"], False)

    bs = train_cfg.get("batch_size", 32)
    dl = lambda d, shuffle: DataLoader(d, batch_size=bs, shuffle=shuffle, num_workers=train_cfg.get("num_workers", 0))
    train_dl, val_dl, test_dl = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    device = _device(train_cfg.get("device", "auto"))
    model = MODELS.build(model_cfg["name"], input_size=val_ds.feature_dim(), num_classes=len(classes),
                         **model_cfg.get("params", {})).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    loss_fn = nn.CrossEntropyLoss(label_smoothing=train_cfg.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.get("lr", 1e-3),
                                  weight_decay=train_cfg.get("weight_decay", 1e-4))
    epochs = train_cfg.get("epochs", 100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    run_dir = (run_root or paths.RUNS) / cfg["name"] / f"fold{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "labels.json").write_text(json.dumps(label_map, indent=1))

    patience, best, bad, history = train_cfg.get("patience", 15), float("inf"), 0, []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr = _run_epoch(model, train_dl, loss_fn, device, optimizer)
        va = _run_epoch(model, val_dl, loss_fn, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["acc"],
                        "val_loss": va["loss"], "val_acc": va["acc"], "lr": optimizer.param_groups[0]["lr"]})
        print(f"[{cfg['name']} fold{fold}] epoch {epoch:3d}  train {tr['loss']:.3f}/{tr['acc']:.3f}  val {va['loss']:.3f}/{va['acc']:.3f}")
        if va["loss"] < best - 1e-4:
            best, bad = va["loss"], 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                break
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    te = _run_epoch(model, test_dl, loss_fn, device)
    metrics = summarize(te["y"], te["probs"], classes)
    best_epoch = min(history, key=lambda h: h["val_loss"])
    metrics.update({"fold": fold, "test_signer": split.get("test_signer"), "params": n_params,
                    "best_epoch": best_epoch["epoch"], "train_acc_at_best": best_epoch["train_acc"],
                    "val_acc_at_best": best_epoch["val_acc"], "train_seconds": round(time.time() - t0, 1)})
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"[{cfg['name']} fold{fold}] TEST acc {metrics['accuracy']:.3f}  macro-F1 {metrics['macro_f1']:.3f}  params {n_params:,}")
    return metrics
