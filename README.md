## Ear-EmoLoad

### Dataset
- **Download**: [Figshare — Ear_Emoload_Dataset](https://figshare.com/articles/dataset/Ear_Emoload_Dataset/31027897?file=60874777)

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
- `ear_emoload_model.py`:  training logic
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

