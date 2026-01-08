from __future__ import annotations

import argparse
import os
from pathlib import Path

from ear_emoload_model import TrainConfig, train_single_subject_conformer
from ear_emoload_dataset import EarEmoLoadFolderDataset, default_read_fn


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ear-emoload: single-subject EEG classification with Conformer.")

    ap.add_argument("--task", type=str, default="emotion", choices=["emotion", "workload"], help="Task name")
    ap.add_argument("--subject", type=int, required=True, help="Single subject id (required)")

    ap.add_argument("--data_root", type=str, default="", help="Dataset root (default: data/ex_data_{task})")
    ap.add_argument("--sfreq", type=float, default=250.0, help="Sampling rate (Hz) for .npy trials")
    ap.add_argument("--window", type=float, default=4.0, help="Window length in seconds")
    ap.add_argument("--step", type=float, default=2.0, help="Window step in seconds")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio by trials")
    ap.add_argument("--seed", type=int, default=43, help="Random seed")

    ap.add_argument("--epochs", type=int, default=50, help="Training epochs")
    ap.add_argument("--batch_size", type=int, default=128, help="Batch size")
    ap.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")

    ap.add_argument("--hid_channels", type=int, default=40)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=10)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--forward_expansion", type=int, default=4)
    ap.add_argument("--forward_dropout", type=float, default=0.5)
    ap.add_argument("--cls_channels", type=int, default=32)
    ap.add_argument("--cls_dropout", type=float, default=0.5)

    ap.add_argument("--run_root", type=str, default="ear_emoload_runs", help="Output root folder")
    ap.add_argument("--run_name", type=str, default="", help="Optional run name (default: auto timestamp)")
    ap.add_argument("--force_reload", action="store_true", help="Rebuild dataset cache")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from torcheeg import transforms  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Missing dependency: torcheeg. Please install dependencies first "
            "(see requirements-pip.txt / environment.yml)."
        ) from e

    data_root = args.data_root.strip() or str(Path("data") / f"ex_data_{args.task}")
    if not os.path.exists(data_root):
        raise FileNotFoundError(
            f"Dataset root not found: {data_root}\n"
            f"Run data export first:\n"
            f"  python ear_emoload_build_data.py --task {args.task} --subjects {args.subject}\n"
        )

    # label_x -> int
    label_map = {f"label_{i}": i for i in range(10)}

    dataset = EarEmoLoadFolderDataset(
        root_path=data_root,
        io_path=str(Path("ear_emoload_cache") / f"{args.task}_sub{args.subject}_w{args.window}_s{args.step}"),
        window_size=float(args.window),
        step_size=float(args.step),
        force_reload=bool(args.force_reload),
        selected_subjects=[int(args.subject)],
        read_fn=lambda fp, window_size, step_size, **kw: default_read_fn(
            fp, window_size=window_size, step_size=step_size, sfreq=float(args.sfreq)
        ),
        online_transform=transforms.Compose([transforms.BaselineRemoval(), transforms.ToTensor()]),
        label_transform=transforms.Compose(
            [transforms.Select("label"), transforms.Lambda(lambda x: label_map.get(str(x), int(str(x).split("_")[-1])))]
        ),
        num_worker=0,
        verbose=True,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"No samples found for subject={args.subject} under: {data_root}\n"
            f"Expected files like: label_0/sub_{args.subject}_session_*/trial_*.npy"
        )

    cfg = TrainConfig(
        task=str(args.task),
        subject=int(args.subject),
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        device=str(args.device),
        sfreq=float(args.sfreq),
        window=float(args.window),
        step=float(args.step),
        hid_channels=int(args.hid_channels),
        depth=int(args.depth),
        heads=int(args.heads),
        dropout=float(args.dropout),
        forward_expansion=int(args.forward_expansion),
        forward_dropout=float(args.forward_dropout),
        cls_channels=int(args.cls_channels),
        cls_dropout=float(args.cls_dropout),
        val_ratio=float(args.val_ratio),
        run_root=str(args.run_root),
        run_name=str(args.run_name),
    )

    out = train_single_subject_conformer(dataset, cfg)
    print(f"[ear-emoload] done. run_dir={out['run_dir']}")
    print(f"[ear-emoload] best_model={out['best_model_path']} best={out['best']}")


if __name__ == "__main__":
    main()


