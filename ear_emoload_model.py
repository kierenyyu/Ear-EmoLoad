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

    # Model selection
    # - ear_embed_cnn: original baseline (EarEmoLoadNet, single-task)
    # - ear_embed_tf_attn: paper-matching Ear-Embed (CWT + ST-CNN + optional MHSA, multi-task)
    model: str = "ear_embed_cnn"

    # Multi-task loss weights (only used when dataset provides dict labels)
    lambda_emo: float = 1.0
    lambda_load: float = 1.0

    # Ablations (only for ear_embed_tf_attn)
    use_attention: bool = True
    mask_band: str = "none"  # none|theta|alpha|beta

    # CWT
    cwt_f_min: float = 1.0
    cwt_f_max: float = 50.0
    cwt_n_freqs: int = 48
    cwt_freqs_scale: str = "log"  # log|linear
    cwt_morlet_w0: float = 6.0
    cwt_out: str = "magnitude"  # magnitude|power

    # ST-CNN
    st_d1: int = 32
    st_d2: int = 64
    st_kt: int = 15
    st_kf: int = 7
    st_pool_t: int = 4
    st_pool_f: int = 2
    st_dropout: float = 0.1

    # Attention / embedding
    attn_num_layers: int = 2
    attn_num_heads: int = 4
    attn_mlp_ratio: float = 4.0
    attn_dropout: float = 0.1
    attn_attn_dropout: float = 0.1
    emb_dim: int = 128
    emb_l2_norm: bool = False

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
        bsz = int(x.size(0))
        # y may be an int tensor (single-task) or a dict (multi-task)
        if isinstance(y, dict):
            # multi-task: y is dict of tensors
            yt = {k: v.to(device).long() for k, v in y.items()}
            out = model(x)
            # prefer model-provided loss if present
            if hasattr(model, "loss"):
                loss = model.loss(out, yt)  # type: ignore[attr-defined]
                # accuracy: report load accuracy as primary (stable 3-way)
                pred = torch.argmax(out["load"], dim=1)
                correct += int((pred == yt["load"]).sum().item())
            else:  # pragma: no cover
                raise RuntimeError("Multi-task model must implement .loss().")
        else:
            y = y.to(device).long()
            logits = model(x)
            loss = loss_fn(logits, y)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
        total_loss += float(loss.item()) * bsz
        n += bsz
    return {"loss": float(total_loss / max(1, n)), "acc": float(correct / max(1, n)), "n": int(n)}


def build_model(dataset, cfg: TrainConfig) -> nn.Module:
    """
    Minimal model factory.
    Keeps original baseline intact.
    """
    num_electrodes = _infer_num_electrodes(dataset)
    if str(cfg.model).strip() in ("ear_embed_cnn", "ear_emoloadnet", "baseline"):
        num_classes = _infer_num_classes(dataset)
        return EarEmoLoadNet(
            num_electrodes=int(num_electrodes),
            num_classes=int(num_classes),
            f1=int(cfg.f1),
            d=int(cfg.d),
            f2=int(cfg.f2),
            kernel_1=int(cfg.kernel_1),
            kernel_2=int(cfg.kernel_2),
            dropout=float(cfg.dropout),
        )

    if str(cfg.model).strip() == "ear_embed_tf_attn":
        from models.ear_embed_tf_attn import (  # local import to keep baseline deps simple
            AttnConfig,
            CWTConfig,
            EarEmbedConfig,
            EarEmbedMultiTask,
            STCNNConfig,
        )

        mcfg = EarEmbedConfig(
            lambda_emo=float(cfg.lambda_emo),
            lambda_load=float(cfg.lambda_load),
            use_attention=bool(cfg.use_attention),
            mask_band=str(cfg.mask_band),
            emb_dim=int(cfg.emb_dim),
            l2_norm=bool(cfg.emb_l2_norm),
            cwt=CWTConfig(
                sfreq=float(cfg.sfreq),
                f_min=float(cfg.cwt_f_min),
                f_max=float(cfg.cwt_f_max),
                n_freqs=int(cfg.cwt_n_freqs),
                freqs_scale=str(cfg.cwt_freqs_scale),
                morlet_w0=float(cfg.cwt_morlet_w0),
                out=str(cfg.cwt_out),
            ),
            stcnn=STCNNConfig(
                d1=int(cfg.st_d1),
                d2=int(cfg.st_d2),
                kt=int(cfg.st_kt),
                kf=int(cfg.st_kf),
                pool_t=int(cfg.st_pool_t),
                pool_f=int(cfg.st_pool_f),
                dropout=float(cfg.st_dropout),
            ),
            attn=AttnConfig(
                num_layers=int(cfg.attn_num_layers),
                num_heads=int(cfg.attn_num_heads),
                mlp_ratio=float(cfg.attn_mlp_ratio),
                dropout=float(cfg.attn_dropout),
                attn_dropout=float(cfg.attn_attn_dropout),
            ),
        )
        return EarEmbedMultiTask(num_electrodes=int(num_electrodes), cfg=mcfg)

    raise ValueError(f"Unknown model={cfg.model!r}. Use ear_embed_cnn or ear_embed_tf_attn.")


def train_single_subject_ear_emoload(dataset, cfg: TrainConfig) -> Dict[str, object]:
    set_seed(int(cfg.seed))

    device = torch.device("cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
    num_electrodes = _infer_num_electrodes(dataset)

    train_ds, val_ds, split_info = split_by_trial_stratified(dataset, val_ratio=cfg.val_ratio, seed=cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=int(cfg.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)

    model = build_model(dataset, cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))

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
            bsz = int(x.size(0))

            opt.zero_grad(set_to_none=True)
            # y may be single-task tensor or dict of tensors (multi-task)
            if isinstance(y, dict):
                yt = {k: v.to(device).long() for k, v in y.items()}
                out = model(x)
                if hasattr(model, "loss"):
                    loss = model.loss(out, yt)  # type: ignore[attr-defined]
                    pred = torch.argmax(out["load"], dim=1)
                    correct += int((pred == yt["load"]).sum().item())
                else:  # pragma: no cover
                    raise RuntimeError("Multi-task model must implement .loss().")
            else:
                y = y.to(device).long()
                logits = model(x)
                loss = nn.CrossEntropyLoss()(logits, y)
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == y).sum().item())
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * bsz
            n += bsz

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
        "num_electrodes": int(num_electrodes),
    }


