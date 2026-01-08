"""
Build `data/ex_data_{task}/label_x/*.npy` from existing per-subject CSV exports.

Current dataset layout:
  data/subject{N}/session_{K}/
    - all_trials_filtered.csv  (columns: trial_{i}_ch{j})
    - trial_metadata.csv       (trial_id, emotion_label, workload_label, data_length, ...)

This script converts each labeled trial into a single `.npy` file:
  (channels, time_points) float32
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("Missing dependency: numpy. Please install requirements-pip.txt.") from e

# Some CSVs may store trial/channel indices as floats in column names (e.g. trial_1.0_ch1).
TRIAL_COL_RE = re.compile(r"^trial_(\d+(?:\.\d+)?)_ch(\d+(?:\.\d+)?)$")


def _read_metadata(metadata_path: Path) -> Dict[int, Dict[str, int]]:
    out: Dict[int, Dict[str, int]] = {}
    with metadata_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                trial_id = int(float(row["trial_id"]))
            except Exception:
                continue

            def _to_int(x, default=-1):
                try:
                    return int(float(x))
                except Exception:
                    return default

            out[trial_id] = {
                "emotion_label": _to_int(row.get("emotion_label", -1), -1),
                "workload_label": _to_int(row.get("workload_label", -1), -1),
                "data_length": _to_int(row.get("data_length", -1), -1),
                "session_id": _to_int(row.get("session_id", -1), -1),
                "subject_id": _to_int(row.get("subject_id", -1), -1),
            }
    return out


def _load_filtered_matrix(filtered_csv: Path) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Returns:
      X: (n_rows, n_cols) float32
      mapping: list[(trial_id, ch)] for each column index
    """
    with filtered_csv.open("r", encoding="utf-8", errors="ignore") as f:
        header_line = f.readline().strip("\n\r")
    header = header_line.split(",")

    mapping: List[Tuple[int, int]] = []
    for col in header:
        m = TRIAL_COL_RE.match(col.strip())
        if not m:
            raise ValueError(f"Unexpected column name: {col!r} in {filtered_csv}")
        trial_id = int(float(m.group(1)))
        ch = int(float(m.group(2)))
        mapping.append((trial_id, ch))

    X = np.genfromtxt(str(filtered_csv), delimiter=",", skip_header=1, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != len(mapping):
        raise ValueError(f"Column count mismatch: parsed={X.shape[1]} header={len(mapping)} for {filtered_csv}")
    return X, mapping


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_for_session(
    subject_dir: Path,
    session_dir: Path,
    out_root: Path,
    task: str,
    skip_label: int = -1,
    dry_run: bool = False,
) -> int:
    meta_path = session_dir / "trial_metadata.csv"
    data_path = session_dir / "all_trials_filtered.csv"
    if not meta_path.exists() or not data_path.exists():
        return 0

    meta = _read_metadata(meta_path)
    X, mapping = _load_filtered_matrix(data_path)

    cols_by_trial: Dict[int, List[Tuple[int, int]]] = {}
    for idx, (trial_id, ch) in enumerate(mapping):
        cols_by_trial.setdefault(trial_id, []).append((ch, idx))

    n_written = 0
    for trial_id, cols in sorted(cols_by_trial.items()):
        info = meta.get(trial_id, None)
        if info is None:
            continue

        label = info["emotion_label"] if task == "emotion" else info["workload_label"]
        if label == skip_label:
            continue

        cols_sorted = [idx for (ch, idx) in sorted(cols, key=lambda t: t[0])]
        trial_mat = X[:, cols_sorted]  # (time, ch)

        dl = int(info.get("data_length", -1))
        if dl > 0 and dl <= trial_mat.shape[0]:
            trial_mat = trial_mat[:dl, :]

        if np.isnan(trial_mat).any():
            mask_all_nan = np.all(np.isnan(trial_mat), axis=1)
            if mask_all_nan.any():
                last_valid = np.where(~mask_all_nan)[0]
                if len(last_valid) > 0:
                    trial_mat = trial_mat[: last_valid[-1] + 1, :]

        trial_mat = np.nan_to_num(trial_mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        arr = trial_mat.T  # (channels, time)

        out_dir = out_root / f"label_{label}"
        _ensure_dir(out_dir)

        sid = info.get("subject_id", -1)
        sess = info.get("session_id", -1)
        out_name = f"sub_{sid}_session_{sess}_trial_{trial_id}.npy"
        out_path = out_dir / out_name
        if not dry_run:
            np.save(out_path, arr)
        n_written += 1
    return n_written


def main() -> None:
    ap = argparse.ArgumentParser(description="ear-emoload: export per-trial .npy files from subject/session CSVs.")
    ap.add_argument("--input_root", type=str, default="data", help="Root containing subject*/session_* folders")
    ap.add_argument("--task", type=str, default="emotion", choices=["emotion", "workload"], help="Which label to export")
    ap.add_argument("--output_root", type=str, default="", help="Output dir (default: data/ex_data_{task})")
    ap.add_argument("--subjects", type=str, default="", help="Comma-separated subject ids (e.g. 1,2,11). Default: all")
    ap.add_argument("--dry_run", action="store_true", help="Do not write files; just report counts")
    args = ap.parse_args()

    in_root = Path(args.input_root)
    if not in_root.exists():
        raise FileNotFoundError(f"input_root not found: {in_root}")

    out_root = Path(args.output_root) if args.output_root else Path("data") / f"ex_data_{args.task}"
    _ensure_dir(out_root)

    selected = None
    if args.subjects.strip():
        selected = {int(x.strip()) for x in args.subjects.split(",") if x.strip()}

    total = 0
    for subject_dir in sorted(in_root.glob("subject*")):
        m = re.match(r"subject(\d+)$", subject_dir.name)
        if not m:
            continue
        sid = int(m.group(1))
        if selected is not None and sid not in selected:
            continue

        for session_dir in sorted(subject_dir.glob("session_*")):
            n = build_for_session(
                subject_dir=subject_dir,
                session_dir=session_dir,
                out_root=out_root,
                task=args.task,
                dry_run=args.dry_run,
            )
            total += n
            if n:
                print(f"[ear-emoload] [{args.task}] {subject_dir.name}/{session_dir.name}: wrote {n} trials")

    print(f"[ear-emoload] Done. Total trials written for task={args.task}: {total}")
    print(f"[ear-emoload] Output: {out_root}")


if __name__ == "__main__":
    main()


