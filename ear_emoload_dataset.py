from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Union

try:
    import mne  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("Missing dependency: mne. Please install requirements-pip.txt.") from e

try:
    import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("Missing dependency: numpy. Please install requirements-pip.txt.") from e
try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("Missing dependency: torch. Please install requirements-pip.txt.") from e
try:
    from torcheeg.datasets.module.base_dataset import BaseDataset  # type: ignore
    from torcheeg.utils import get_random_dir_path  # type: ignore
except Exception:  # pragma: no cover
    BaseDataset = object  # type: ignore
    get_random_dir_path = None  # type: ignore


class _NumpyEpochs:
    """
    A tiny epochs-like wrapper compatible with this project's `process_record`.

    It cuts one (channels, time) or (time, channels) trial into fixed-length windows.
    """

    def __init__(self, data: np.ndarray, sfreq: float, window_size: float, step_size: float):
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D array, got shape={getattr(data, 'shape', None)}")

        # data: (channels, time) or (time, channels)
        x = data if data.shape[0] <= data.shape[1] else data.T
        n_ch, n_t = x.shape

        self.info = {"sfreq": float(sfreq)}
        win = int(round(float(window_size) * float(sfreq)))
        step = int(round(float(step_size) * float(sfreq)))
        if win <= 0 or step <= 0:
            raise ValueError(f"Invalid window/step: window={window_size}, step={step_size}, sfreq={sfreq}")

        if n_t < win:
            self._data = np.empty((0, n_ch, win), dtype=np.float32)
            self.events = np.zeros((0, 3), dtype=int)
            return

        starts = list(range(0, n_t - win + 1, step))
        self.events = np.zeros((len(starts), 3), dtype=int)
        self.events[:, 0] = np.asarray(starts, dtype=int)
        self._data = np.stack([x[:, s : s + win] for s in starts], axis=0).astype(np.float32, copy=False)

    def get_data(self) -> np.ndarray:
        return self._data


def default_read_fn(file_path, window_size=4.0, step_size=2.0, **kwargs):
    """
    Read one trial file (.edf or .npy) and apply a sliding window.

    Returns an object with:
    - .info["sfreq"]
    - .events[:, 0] containing window start sample indices
    - .get_data() returning (n_windows, channels, window_samples)
    """

    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].lower()

    # 1) NPY: one trial per file, shape (channels,time) or (time,channels)
    if ext == ".npy":
        data = np.load(file_path, allow_pickle=False)
        sfreq = float(kwargs.get("sfreq", 250.0))
        return _NumpyEpochs(data=data, sfreq=sfreq, window_size=window_size, step_size=step_size)

    # 2) EDF
    raw = mne.io.read_raw(file_path, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    overlap = float(window_size) - float(step_size)
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=float(window_size),
        overlap=float(overlap),
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    return epochs


class EarEmoLoadFolderDataset(BaseDataset):
    """
    Read EEG trials from a folder tree like:

      data_root/
        label_0/
          sub_{subject}_session_{session}_trial_{trial}.npy
        label_1/
          ...

    Supported trial formats: `.npy` and `.edf`.
    """

    def __init__(
        self,
        root_path: str,
        io_path: Union[None, str] = None,
        io_size: int = 1048576,
        io_mode: str = "lmdb",
        num_worker: int = 0,
        verbose: bool = True,
        read_fn: Union[None, Callable] = default_read_fn,
        online_transform: Union[None, Callable] = None,
        offline_transform: Union[None, Callable] = None,
        label_transform: Union[None, Callable] = None,
        window_size: float = 4.0,
        step_size: float = 2.0,
        force_reload: bool = False,
        selected_subjects: Union[None, list[int]] = None,
        **kwargs,
    ):
        if get_random_dir_path is None:
            raise ImportError(
                "Missing dependency: torcheeg. Please install dependencies first "
                "(see requirements-pip.txt / environment.yml)."
            )
        self.window_size = float(window_size)
        self.step_size = float(step_size)
        self.force_reload = bool(force_reload)

        if io_path is None:
            io_path = get_random_dir_path(dir_prefix="datasets")

        def _safe_rmtree_io_path(path_str: str) -> None:
            """
            Remove io_path only when it is inside the project directory.
            This prevents accidental removal of unrelated paths.
            """
            import shutil

            p = Path(path_str).expanduser()
            try:
                p_resolved = p.resolve()
            except Exception:
                p_resolved = p.absolute()

            project_root = Path(__file__).resolve().parent
            if not (p_resolved == project_root or project_root in p_resolved.parents):
                raise RuntimeError(
                    f"Refusing to remove io_path outside the project: {path_str} (resolved: {p_resolved})"
                )
            if p_resolved.exists() and p_resolved.is_dir():
                shutil.rmtree(p_resolved)

        params = {
            "root_path": root_path,
            "structure": "subject_in_label",
            "read_fn": read_fn,
            "online_transform": online_transform,
            "offline_transform": offline_transform,
            "label_transform": label_transform,
            "io_path": io_path,
            "io_size": io_size,
            "io_mode": io_mode,
            "num_worker": num_worker,
            "verbose": verbose,
            "window_size": self.window_size,
            "step_size": self.step_size,
        }
        params.update(kwargs)

        if self.force_reload and os.path.exists(io_path):
            if verbose:
                print(f"[ear-emoload] Rebuilding dataset cache: {io_path}")
            _safe_rmtree_io_path(io_path)
        elif verbose:
            if os.path.exists(io_path) and os.path.isdir(io_path) and len(os.listdir(io_path)) > 0:
                print(f"[ear-emoload] Using dataset cache: {io_path}")
            else:
                print(f"[ear-emoload] Creating new dataset cache: {io_path}")

        try:
            super().__init__(**params)
        except AssertionError as e:
            msg = str(e)
            if ("io_path" in msg) and ("is corrupted" in msg):
                if verbose:
                    print(f"[ear-emoload] Cache appears corrupted; rebuilding: {io_path}")
                _safe_rmtree_io_path(io_path)
                super().__init__(**params)
            else:
                raise

        self.__dict__.update(params)
        self.selected_subjects = selected_subjects

    @staticmethod
    def process_record(
        file: Any = None,
        offline_transform: Union[None, Callable] = None,
        read_fn: Union[None, Callable] = None,
        window_size: float = 4.0,
        step_size: float = 2.0,
        **kwargs,
    ):
        file_path, subject_id, session_id, label, trial_id = file
        epochs = read_fn(file_path, window_size=window_size, step_size=step_size, **kwargs)

        sfreq = float(epochs.info["sfreq"])
        window_samples = int(float(window_size) * sfreq)

        write_pointer = 0
        for i, trial_signal in enumerate(epochs.get_data()):
            t_eeg = trial_signal
            if offline_transform is not None:
                t = offline_transform(eeg=trial_signal)
                t_eeg = t["eeg"]

            clip_id = f"{subject_id}_{session_id}_{label}_{write_pointer}"
            write_pointer += 1

            start_at = int(epochs.events[i, 0])
            end_at = int(start_at + window_samples)

            record_info = {
                "subject_id": subject_id,
                "session_id": session_id,
                "start_at": start_at,
                "end_at": end_at,
                "clip_id": clip_id,
                "label": label,
                "trial_id": trial_id,
            }

            yield {"eeg": t_eeg, "key": clip_id, "info": record_info}

    def set_records(self, root_path: str = "./folder", structure: str = "subject_in_label", **kwargs):
        selected_subjects = kwargs.get("selected_subjects", None)

        subfolders = [str(i) for i in Path(root_path).iterdir() if i.is_dir()]
        file_path_list = []
        for subfolder in subfolders:
            for p in Path(subfolder).iterdir():
                if p.is_file() and p.suffix.lower() in (".edf", ".npy"):
                    file_path_list.append(str(p))

        if structure != "subject_in_label":
            raise ValueError("EarEmoLoadFolderDataset only supports structure='subject_in_label'.")

        subjects, session_ids, trial_ids, filtered_file_paths = [], [], [], []
        for file_path in file_path_list:
            base = os.path.basename(file_path)
            file_name = os.path.splitext(base)[0]
            parts = file_name.split("_")
            if len(parts) >= 6 and parts[0] == "sub" and parts[2] == "session" and parts[4] == "trial":
                subject = parts[1]
                if selected_subjects is not None and int(subject) not in selected_subjects:
                    continue
                session_id = parts[3]
                trial_id = parts[5]
                subjects.append(subject)
                session_ids.append(session_id)
                trial_ids.append(trial_id)
                filtered_file_paths.append(file_path)

        file_path_list = filtered_file_paths
        labels = [Path(i).parent.name for i in file_path_list]

        return list(zip(file_path_list, subjects, session_ids, labels, trial_ids))


class EarEmoLoadMultiTaskWindowDataset:
    """
    Multi-task dataset that aligns emotion + workload labels by trial file name.

    It expects exported folders:
      - emotion_root:   data/ex_data_emotion/label_k/sub_{sid}_session_{sess}_trial_{trial}.npy
      - workload_root:  data/ex_data_workload/label_k/sub_{sid}_session_{sess}_trial_{trial}.npy

    Each __getitem__ returns:
      x: FloatTensor (C, T)
      y: dict {"valence": int, "load": int}

    Notes:
    - We keep this as a lightweight torch Dataset (not torcheeg BaseDataset) to avoid invasive refactors.
    - For speed, we keep a small in-memory cache of the most recently used trial windows.
    """

    def __init__(
        self,
        emotion_root: str,
        workload_root: str,
        sfreq: float = 250.0,
        window_size: float = 4.0,
        step_size: float = 2.0,
        selected_subjects: Union[None, list[int]] = None,
        max_trial_cache: int = 8,
        verbose: bool = True,
    ):
        self.emotion_root = str(emotion_root)
        self.workload_root = str(workload_root)
        self.sfreq = float(sfreq)
        self.window_size = float(window_size)
        self.step_size = float(step_size)
        self.selected_subjects = selected_subjects
        self.max_trial_cache = int(max_trial_cache)
        self.verbose = bool(verbose)

        self._pairs = self._index_pairs()
        # Flatten windows across trials: each entry points to (trial_idx, window_idx)
        self._flat_index = self._build_flat_index()

        # Mimic `.info` used by split_by_trial_stratified in training code
        self.info = []
        for trial_idx, row in enumerate(self._pairs):
            trial_id = row["trial_id"]
            # combined label for stratification: emo*10 + load (robust for small class counts)
            comb = int(row["emotion_label"]) * 10 + int(row["workload_label"])
            self.info.append({"trial_id": trial_id, "label": comb, "trial_idx": trial_idx})

        # trial window cache: trial_idx -> numpy array (n_win, C, T)
        self._trial_cache = {}  # type: ignore[var-annotated]
        self._trial_cache_order = []  # type: ignore[var-annotated]

        if self.verbose:
            print(
                f"[ear-emoload] MultiTaskDataset indexed {len(self._pairs)} trials -> "
                f"{len(self._flat_index)} windows (w={self.window_size}s step={self.step_size}s)"
            )

    def _index_pairs(self):
        from pathlib import Path

        def _scan(root: str):
            rootp = Path(root)
            if not rootp.exists():
                raise FileNotFoundError(f"Dataset root not found: {root}")
            out = {}
            for label_dir in sorted(rootp.glob("label_*")):
                if not label_dir.is_dir():
                    continue
                try:
                    lab = int(label_dir.name.split("_")[-1])
                except Exception:
                    continue
                for fp in label_dir.glob("*.npy"):
                    out[fp.name] = {"path": str(fp), "label": lab}
            return out

        emo = _scan(self.emotion_root)
        wkl = _scan(self.workload_root)
        common = sorted(set(emo.keys()) & set(wkl.keys()))
        if not common:
            raise RuntimeError(
                "No overlapping trials between emotion_root and workload_root. "
                "Ensure both exports were generated for the same subjects/sessions."
            )

        pairs = []
        for name in common:
            # Parse subject/session/trial from filename: sub_{sid}_session_{sess}_trial_{trial}.npy
            base = name.replace(".npy", "")
            parts = base.split("_")
            sid = None
            sess = None
            tid = None
            if len(parts) >= 6 and parts[0] == "sub" and parts[2] == "session" and parts[4] == "trial":
                try:
                    sid = int(parts[1])
                    sess = int(parts[3])
                    tid = int(parts[5])
                except Exception:
                    sid, sess, tid = None, None, None

            if self.selected_subjects is not None and sid is not None and int(sid) not in self.selected_subjects:
                continue

            pairs.append(
                {
                    "file_name": name,
                    "trial_id": str(tid) if tid is not None else base,
                    "subject_id": sid,
                    "session_id": sess,
                    "path": emo[name]["path"],  # use emotion path for signal (identical across exports)
                    "emotion_label": int(emo[name]["label"]),
                    "workload_label": int(wkl[name]["label"]),
                }
            )
        if not pairs:
            raise RuntimeError("No trials left after filtering selected_subjects.")
        return pairs

    def _build_flat_index(self):
        # Determine number of windows per trial by reading only metadata once per trial (cheap for npy).
        flat = []
        for trial_idx, row in enumerate(self._pairs):
            epochs = default_read_fn(
                row["path"], window_size=self.window_size, step_size=self.step_size, sfreq=self.sfreq
            )
            n_win = int(epochs.get_data().shape[0])
            for w in range(n_win):
                flat.append((trial_idx, w))
        return flat

    def __len__(self) -> int:
        return int(len(self._flat_index))

    def _get_trial_windows(self, trial_idx: int):
        import numpy as np

        if trial_idx in self._trial_cache:
            # refresh LRU order
            if trial_idx in self._trial_cache_order:
                self._trial_cache_order.remove(trial_idx)
            self._trial_cache_order.append(trial_idx)
            return self._trial_cache[trial_idx]

        row = self._pairs[int(trial_idx)]
        epochs = default_read_fn(row["path"], window_size=self.window_size, step_size=self.step_size, sfreq=self.sfreq)
        windows = epochs.get_data().astype(np.float32, copy=False)  # (n_win, C, T)

        # LRU insert
        self._trial_cache[trial_idx] = windows
        self._trial_cache_order.append(trial_idx)
        while len(self._trial_cache_order) > int(self.max_trial_cache):
            old = self._trial_cache_order.pop(0)
            self._trial_cache.pop(old, None)
        return windows

    def __getitem__(self, idx: int):
        trial_idx, win_idx = self._flat_index[int(idx)]
        row = self._pairs[int(trial_idx)]
        windows = self._get_trial_windows(int(trial_idx))
        x = torch.from_numpy(windows[int(win_idx)])  # (C,T)
        y = {"valence": int(row["emotion_label"]), "load": int(row["workload_label"])}
        return x, y


