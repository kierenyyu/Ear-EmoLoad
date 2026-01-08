## ear-emoload


### Repository layout
- `ear_emoload_build_data.py`: export trials from CSV to `.npy`
- `ear_emoload_dataset.py`: dataset + sliding-window loader
- `ear_emoload_model.py`: Conformer training logic
- `ear_emoload.py`: CLI entrypoint
- `docs/`: extended documentation

### Install

#### Option A: pip

```bash
pip install -r requirements-pip.txt
```

#### Option B: conda

```bash
conda env create -f environment.yml
conda activate ear-emoload
```

### Data format
After export, trials are stored as:

`data/ex_data_{task}/label_{k}/sub_{subject}_session_{session}_trial_{trial}.npy`

Each `.npy` is a float32 array of shape `(channels, time_points)`.

See `docs/DATA.md` for details.

### Quickstart

From the repository root:

```bash
cd /data/user/kyu219/WORKSPACE/Project/EAR_EMOLOAD
```

#### Step 1: export trials

```bash
python ear_emoload_build_data.py --task emotion --subjects 2
```

#### Step 2: train (single subject)

```bash
python ear_emoload.py --task emotion --subject 2 --window 4 --step 2 --sfreq 250 --epochs 50
```

### Outputs
Outputs are written to:

`ear_emoload_runs/sub_{subject}/{task}/ear-emoload_.../`

