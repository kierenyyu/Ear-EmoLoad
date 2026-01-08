## Ear-EmoLoad

**Ear-EmoLoad** is an ear-EEG dataset for studying emotion and cognitive load in naturalistic listening and learning scenarios. The dataset is collected using a custom ear-worn EEG system embedded in in-ear headphones, enabling unobtrusive neural sensing during everyday tasks.

Participants engaged in audio-visual learning and listening activities designed to elicit variations in emotional valence and cognitive load. The dataset supports joint investigation of affective responses and mental workload in an e-learning–like context.

Ear-EmoLoad primarily includes multi-channel ear-EEG recordings with synchronized task markers and annotations. Each data segment is labeled with emotion-related and cognitive load information, along with subject-level metadata. Preprocessing steps are applied to ensure signal quality suitable for ear-EEG analysis.

This dataset is intended for research in affective computing, cognitive load assessment, wearable brain–computer interfaces, and EEG representation learning.

All data were collected with informed consent and are anonymized for research use only.

### Dataset
- **Download**: [Figshare — Ear_Emoload_Dataset](https://figshare.com/articles/dataset/Ear_Emoload_Dataset/31027897?file=60874777)

### This repository
This repository provides a minimal reference pipeline to:
- Export per-trial `.npy` files from the released CSV format
- Train a **Conformer** model for **single-subject** classification

### Installation

#### Option A: pip

```bash
pip install -r requirements-pip.txt
```

#### Option B: conda

```bash
conda env create -f environment.yml
conda activate ear-emoload
```

### Project structure
- `ear_emoload_build_data.py`: export trials from CSV to `.npy`
- `ear_emoload_dataset.py`: dataset + sliding-window loader
- `ear_emoload_model.py`: Conformer training logic
- `ear_emoload.py`: CLI entrypoint
- `docs/`: extended documentation (`docs/DATA.md`, `docs/TRAINING.md`)

### Data format (exported)
Trials are stored as:

`data/ex_data_{task}/label_{k}/sub_{subject}_session_{session}_trial_{trial}.npy`

Each `.npy` is a float32 array of shape `(channels, time_points)`.

### Quickstart

```bash
cd /data/user/kyu219/WORKSPACE/Project/EAR_EMOLOAD
```

#### 1) Export trials

```bash
python ear_emoload_build_data.py --task emotion --subjects 2
```

#### 2) Train (single subject)

```bash
python ear_emoload.py --task emotion --subject 2 --window 4 --step 2 --sfreq 250 --epochs 50
```

### Outputs
Each run writes to:

`ear_emoload_runs/sub_{subject}/{task}/ear-emoload_.../`

### Troubleshooting
- If you see an error like "Missing dependency: torcheeg", install dependencies via `requirements-pip.txt` or `environment.yml`.


