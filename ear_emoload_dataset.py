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
            Delete io_path only when it is inside the project directory.
            This prevents accidental deletion of unrelated paths.
            """
            import shutil

            p = Path(path_str).expanduser()
            try:
                p_resolved = p.resolve()
            except Exception:
                p_resolved = p.absolute()

            project_root = Path(__file__).resolve().parent
            if not (p_resolved == project_root or project_root in p_resolved.parents):
                raise RuntimeError(f"Refusing to delete io_path outside the project: {path_str} (resolved: {p_resolved})")
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


