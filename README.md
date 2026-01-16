# Blackjack YOLO
**Realtime card detection + on-screen blackjack decision (GPU-accelerated)**

This project uses **Ultralytics YOLO** to detect playing card ranks (`A–K`) in real time and optionally expose predictions via webcam or FastAPI.

---

## 0) Prerequisites (IMPORTANT)

### Hardware
- NVIDIA GPU (tested with **RTX 3060 Ti**)
- Updated NVIDIA drivers (`nvidia-smi` must work)

### Software
- **Python 3.12** (required for CUDA-enabled PyTorch)
- Windows PowerShell or Terminal

> ⚠️ Python 3.13 will **NOT** work with CUDA PyTorch (CPU-only).

---

## 1) Install Python 3.12 (Windows)

Download and install from:  
https://www.python.org/downloads/release/python-3120/

During install:
- ✅ Add Python to PATH  
- ✅ Install for all users  

Verify:
```powershell
py -3.12 --version
```

---

## 2) Create & activate a virtual environment

From the project root:

```powershell
py -3.12 -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

If PowerShell blocks activation, run **once**:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Verify:
```powershell
python --version
```

Expected:
```
Python 3.12.x
```

---

## 3) Install dependencies (GPU-enabled)

### Install CUDA-enabled PyTorch (IMPORTANT)
PyTorch is **NOT** installed via `requirements.txt` to avoid CPU-only installs.

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU works:
```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected:
```
True
NVIDIA GeForce RTX ...
```

---

### Install project dependencies

```powershell
pip install -r requirements.txt
```

---

## 4) Dataset structure

Place your labeled dataset like this:

```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Classes (rank-only)

```
A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K
```

### `dataset/data.yaml` (IMPORTANT)

Class names **must be strings**:

```yaml
path: dataset

train: train/images
val: val/images
test: test/images

nc: 13
names: ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
```

> ⚠️ Unquoted numbers in YAML become integers and break labeling, logging, and exports.

---

## 5) Train (GPU)

Run:
```powershell
python train.py
```

Your `train.py` should include:
```python
device=0  # use GPU
```

Expected output:
- `GPU_mem` > 0
- Much faster training than CPU

Weights are saved to:
```
runs/train/<run_name>/weights/best.pt
```

---

## 6) Webcam (real-time detection + decision)

```powershell
python webcam.py
```

- Shows bounding boxes + blackjack decision
- Press **Q** to quit

---

## 7) FastAPI (optional)

Start API:
```powershell
uvicorn api_fastapi:app --reload --port 8000
```

Endpoints:
- `GET /health`
- `POST /infer` (multipart form file)

---

## Common pitfalls

- ❌ Using Python 3.13 → CUDA will NOT work  
- ❌ Forgetting to activate `venv`
- ❌ Installing PyTorch without CUDA
- ❌ `GPU_mem 0G` → CPU-only PyTorch
- ❌ Unquoted class names in `data.yaml`
- ❌ Label indices not in range `0–12`