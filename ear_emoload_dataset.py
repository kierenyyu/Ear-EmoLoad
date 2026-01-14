from __future__ import annotations

import argparse
import os
from pathlib import Path

from ear_emoload_model import TrainConfig, train_single_subject_ear_emoload
from ear_emoload_dataset import EarEmoLoadFolderDataset, EarEmoLoadMultiTaskWindowDataset, default_read_fn


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ear-emoload: single-subject ear-EEG classification (EarEmoLoadNet).")

    ap.add_argument(
        "--task",
        type=str,
        default="emotion",
        choices=["emotion", "workload", "multitask"],
        help="Task name (multitask aligns emotion+workload by trial file name)",
    )
    ap.add_argument("--subject", type=int, required=True, help="Single subject id (required)")

    ap.add_argument("--data_root", type=str, default="", help="Dataset root (default: data/ex_data_{task})")
    ap.add_argument(
        "--emotion_root",
        type=str,
        default="",
        help="For --task multitask: emotion export root (default: data/ex_data_emotion)",
    )
    ap.add_argument(
        "--workload_root",
        type=str,
        default="",
        help="For --task multitask: workload export root (default: data/ex_data_workload)",
    )
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

    ap.add_argument("--f1", type=int, default=16, help="EarEmoLoadNet: temporal filter count")
    ap.add_argument("--d", type=int, default=2, help="EarEmoLoadNet: depth multiplier")
    ap.add_argument("--f2", type=int, default=32, help="EarEmoLoadNet: projection channels")
    ap.add_argument("--kernel_1", type=int, default=64, help="EarEmoLoadNet: temporal kernel size (stage 1)")
    ap.add_argument("--kernel_2", type=int, default=16, help="EarEmoLoadNet: temporal kernel size (stage 2)")
    ap.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")

    ap.add_argument(
        "--model",
        type=str,
        default="ear_embed_cnn",
        choices=["ear_embed_cnn", "ear_embed_tf_attn"],
        help="Model name (ear_embed_cnn keeps original baseline; ear_embed_tf_attn matches paper figure)",
    )
    ap.add_argument("--use_attention", type=int, default=1, help="Ablation: 1=use MHSA, 0=remove attention (Table 4A)")
    ap.add_argument(
        "--mask_band",
        type=str,
        default="none",
        choices=["none", "theta", "alpha", "beta"],
        help="Ablation: mask frequency band on TF maps (Table 4B)",
    )
    ap.add_argument("--lambda_emo", type=float, default=1.0, help="Multi-task CE weight for valence")
    ap.add_argument("--lambda_load", type=float, default=1.0, help="Multi-task CE weight for cognitive load")

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

    if args.task == "multitask":
        emo_root = args.emotion_root.strip() or str(Path("data") / "ex_data_emotion")
        wkl_root = args.workload_root.strip() or str(Path("data") / "ex_data_workload")
        if not os.path.exists(emo_root) or not os.path.exists(wkl_root):
            raise FileNotFoundError(
                "Multi-task requires both exports:\n"
                f"- emotion_root={emo_root}\n"
                f"- workload_root={wkl_root}\n"
                "Run data export first:\n"
                f"  python ear_emoload_build_data.py --task emotion --subjects {args.subject}\n"
                f"  python ear_emoload_build_data.py --task workload --subjects {args.subject}\n"
            )

        dataset = EarEmoLoadMultiTaskWindowDataset(
            emotion_root=emo_root,
            workload_root=wkl_root,
            sfreq=float(args.sfreq),
            window_size=float(args.window),
            step_size=float(args.step),
            selected_subjects=[int(args.subject)],
            verbose=True,
        )
    else:
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
        f1=int(args.f1),
        d=int(args.d),
        f2=int(args.f2),
        kernel_1=int(args.kernel_1),
        kernel_2=int(args.kernel_2),
        dropout=float(args.dropout),
        model=str(args.model),
        use_attention=bool(int(args.use_attention)),
        mask_band=str(args.mask_band),
        lambda_emo=float(args.lambda_emo),
        lambda_load=float(args.lambda_load),
        val_ratio=float(args.val_ratio),
        run_root=str(args.run_root),
        run_name=str(args.run_name),
    )

    out = train_single_subject_ear_emoload(dataset, cfg)
    print(f"[ear-emoload] done. run_dir={out['run_dir']}")
    print(f"[ear-emoload] best_model={out['best_model_path']} best={out['best']}")


if __name__ == "__main__":
    main()


