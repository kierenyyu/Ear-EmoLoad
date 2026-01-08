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

    # EarEmoLoad model hyper-params
    f1: int = 16
    d: int = 2
    f2: int = 32
    kernel_1: int = 64
    kernel_2: int = 16
    dropout: float = 0.5

    # Split
    val_ratio: float = 0.2

    # Output
    run_root: str = "ear_emoload_runs"
    run_name: str = ""  # if empty -> auto timestamp


class EarEmoLoadNet(nn.Module):
    """
    A lightweight ear-EEG classifier.

    Input:  x of shape (B, C, T)
    Output: logits of shape (B, num_classes)

    The network uses temporal filtering + depthwise spatial filtering and global pooling,
    so it works with arbitrary window length T.
    """

    def __init__(
        self,
        num_electrodes: int,
        num_classes: int,
        f1: int = 16,
        d: int = 2,
        f2: int = 32,
        kernel_1: int = 64,
        kernel_2: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()
        if num_electrodes <= 0:
            raise ValueError(f"num_electrodes must be > 0, got {num_electrodes}")
        if num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")

        f1 = int(f1)
        d = int(d)
        f2 = int(f2)
        kernel_1 = int(kernel_1)
        kernel_2 = int(kernel_2)

        self.num_electrodes = int(num_electrodes)
        self.num_classes = int(num_classes)

        # (B, 1, C, T)
        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, kernel_1), padding=(0, kernel_1 // 2), bias=False),
            nn.BatchNorm2d(f1),
        )

        # depthwise spatial conv over electrodes
        self.spatial = nn.Sequential(
            nn.Conv2d(f1, f1 * d, kernel_size=(self.num_electrodes, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(inplace=True),
            nn.Dropout(float(dropout)),
        )

        # separable temporal conv
        self.separable = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, kernel_size=(1, kernel_2), padding=(0, kernel_2 // 2), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(inplace=True),
            nn.Dropout(float(dropout)),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(f2, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B,C,T), got {tuple(x.shape)}")
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.temporal(x)
        x = self.spatial(x)  # (B, f1*d, 1, T)
        x = self.separable(x)  # (B, f2, 1, T)
        x = self.pool(x)  # (B, f2, 1, 1)
        x = x.flatten(1)  # (B, f2)
        return self.classifier(x)


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


def train_single_subject_ear_emoload(dataset, cfg: TrainConfig) -> Dict[str, object]:
    set_seed(int(cfg.seed))

    device = torch.device("cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
    num_classes = _infer_num_classes(dataset)
    num_electrodes = _infer_num_electrodes(dataset)

    train_ds, val_ds, split_info = split_by_trial_stratified(dataset, val_ratio=cfg.val_ratio, seed=cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=int(cfg.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)

    model = EarEmoLoadNet(
        num_electrodes=int(num_electrodes),
        num_classes=int(num_classes),
        f1=int(cfg.f1),
        d=int(cfg.d),
        f2=int(cfg.f2),
        kernel_1=int(cfg.kernel_1),
        kernel_2=int(cfg.kernel_2),
        dropout=float(cfg.dropout),
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


