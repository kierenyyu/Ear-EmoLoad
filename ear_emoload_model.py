from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("Missing dependency: numpy. Please install requirements-pip.txt.") from e

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("Missing dependency: torch. Please install requirements-pip.txt.") from e
from torch import nn
from torch.utils.data import DataLoader, Subset


@dataclass
class TrainConfig:
    task: str
    subject: int
    seed: int = 43
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    device: str = "cuda"

    # Dataset windowing
    sfreq: float = 250.0
    window: float = 4.0
    step: float = 2.0

    # Model hyper-params
    hid_channels: int = 40
    depth: int = 6
    heads: int = 10
    dropout: float = 0.5
    forward_expansion: int = 4
    forward_dropout: float = 0.5
    cls_channels: int = 32
    cls_dropout: float = 0.5

    # Split
    val_ratio: float = 0.2

    # Output
    run_root: str = "ear_emoload_runs"
    run_name: str = ""  # if empty -> auto timestamp


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _infer_num_classes(dataset) -> int:
    labels = []
    for i in range(min(len(dataset), 2048)):
        _, y = dataset[i]
        labels.append(int(y))
    if not labels:
        raise ValueError("Dataset is empty; cannot infer num_classes.")
    return int(max(labels) + 1)


def _infer_num_electrodes(dataset) -> int:
    x, _ = dataset[0]
    if not hasattr(x, "shape"):
        raise ValueError("Unexpected sample type; cannot infer num_electrodes.")
    # expected (channels, time)
    return int(x.shape[0])


def split_by_trial_stratified(dataset, val_ratio: float, seed: int) -> Tuple[Subset, Subset, Dict[str, int]]:
    """
    Split windows by trial_id, so windows from the same trial never leak into both train and val.
    Assumes each trial has a single label (true for this dataset naming scheme).
    """
    if not hasattr(dataset, "info"):
        raise ValueError("Dataset must expose .info (list of dicts).")

    trial_to_indices: Dict[str, List[int]] = {}
    trial_to_label: Dict[str, int] = {}
    for idx, row in enumerate(dataset.info):
        tid = str(row.get("trial_id", "unknown"))
        lab = int(row.get("label", -1))
        trial_to_indices.setdefault(tid, []).append(idx)
        if tid not in trial_to_label:
            trial_to_label[tid] = lab

    trials = sorted(trial_to_indices.keys())
    if not trials:
        raise ValueError("No trials found in dataset.info.")

    trial_labels = np.asarray([trial_to_label[t] for t in trials], dtype=int)

    rng = np.random.RandomState(int(seed))
    uniq = np.unique(trial_labels)
    can_stratify = all(np.sum(trial_labels == u) >= 2 for u in uniq) and len(uniq) >= 2

    n_val_trials = max(1, int(round(len(trials) * float(val_ratio))))
    if n_val_trials >= len(trials):
        n_val_trials = max(1, len(trials) - 1)

    if can_stratify:
        val_trials: List[str] = []
        for u in uniq:
            cls_trials = [t for t, y in zip(trials, trial_labels) if y == u]
            rng.shuffle(cls_trials)
            take = max(1, int(round(len(cls_trials) * float(val_ratio))))
            val_trials.extend(cls_trials[:take])
        val_trials = sorted(set(val_trials))
        if len(val_trials) >= len(trials):
            rng.shuffle(trials)
            val_trials = trials[:n_val_trials]
    else:
        rng.shuffle(trials)
        val_trials = trials[:n_val_trials]

    val_set = set(val_trials)
    train_indices: List[int] = []
    val_indices: List[int] = []
    for t, idxs in trial_to_indices.items():
        if t in val_set:
            val_indices.extend(idxs)
        else:
            train_indices.extend(idxs)

    split_info = {
        "n_trials": int(len(trials)),
        "n_train_trials": int(len(trials) - len(val_set)),
        "n_val_trials": int(len(val_set)),
        "n_train_windows": int(len(train_indices)),
        "n_val_windows": int(len(val_indices)),
    }
    return Subset(dataset, train_indices), Subset(dataset, val_indices), split_info


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long()
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * int(y.size(0))
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        n += int(y.size(0))
    return {"loss": float(total_loss / max(1, n)), "acc": float(correct / max(1, n)), "n": int(n)}


def train_single_subject_conformer(dataset, cfg: TrainConfig) -> Dict[str, object]:
    try:
        from torcheeg.models import Conformer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Missing dependency: torcheeg. Please install dependencies first "
            "(see requirements-pip.txt / environment.yml)."
        ) from e

    set_seed(int(cfg.seed))

    device = torch.device("cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
    num_classes = _infer_num_classes(dataset)
    num_electrodes = _infer_num_electrodes(dataset)

    if cfg.hid_channels % cfg.heads != 0:
        for h in range(int(cfg.heads), 0, -1):
            if cfg.hid_channels % h == 0:
                cfg.heads = int(h)
                break

    train_ds, val_ds, split_info = split_by_trial_stratified(dataset, val_ratio=cfg.val_ratio, seed=cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=int(cfg.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)

    model = Conformer(
        num_electrodes=int(num_electrodes),
        sampling_rate=float(cfg.sfreq),
        hid_channels=int(cfg.hid_channels),
        depth=int(cfg.depth),
        heads=int(cfg.heads),
        dropout=float(cfg.dropout),
        forward_expansion=int(cfg.forward_expansion),
        forward_dropout=float(cfg.forward_dropout),
        cls_channels=int(cfg.cls_channels),
        cls_dropout=float(cfg.cls_dropout),
        num_classes=int(num_classes),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    loss_fn = nn.CrossEntropyLoss()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name.strip() or f"ear-emoload_{cfg.task}_sub{cfg.subject}_seed{cfg.seed}_{ts}"
    run_dir = Path(cfg.run_root) / f"sub_{cfg.subject}" / cfg.task / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, float]] = []
    best = {"val_acc": -1.0, "epoch": -1}
    best_path = run_dir / "best_model.pth"

    t0 = time.time()
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).long()

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * int(y.size(0))
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            n += int(y.size(0))

        train_loss = float(total_loss / max(1, n))
        train_acc = float(correct / max(1, n))
        val_metrics = evaluate(model, val_loader, device=device)

        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["acc"]),
        }
        history.append(row)

        print(
            f"[ear-emoload] epoch {epoch:03d}/{cfg.epochs} | "
            f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} | "
            f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.4f}"
        )

        if row["val_acc"] > best["val_acc"]:
            best = {"val_acc": float(row["val_acc"]), "epoch": int(epoch)}
            torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)}, best_path)

    elapsed = float(time.time() - t0)

    torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)}, run_dir / "last_model.pth")
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "project": "ear-emoload",
                "task": cfg.task,
                "subject": int(cfg.subject),
                "seed": int(cfg.seed),
                "split": split_info,
                "best": best,
                "history": history,
                "elapsed_sec": elapsed,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=True, indent=2)

    return {
        "run_dir": str(run_dir),
        "best_model_path": str(best_path),
        "best": best,
        "split": split_info,
        "elapsed_sec": elapsed,
        "num_classes": int(num_classes),
        "num_electrodes": int(num_electrodes),
    }


