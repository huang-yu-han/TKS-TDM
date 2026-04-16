# TKS-TDM: Temporal Key Selection-Transformer Diagnostic Model

**Official implementation of the paper:**

> **TKS-TDM: A Temporal Key Point Selection Model for Efficient Fault Diagnosis of Railway Turnout Actuators Using Multi-Sensor Signals**
> *Yuhan Huang, Xiaoxi Hu, Yiming He, Jingming Cao, Tao Tang, Weiming Shen*
> IEEE Transactions on Transportation Electrification, 2025

[![Paper](https://img.shields.io/badge/Paper-IEEE_TTE-blue)](https://doi.org/10.1109/TTE.2025.XXXXXX)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-orange)](https://pytorch.org/)

---

## Overview

Railway turnout actuators (RTAs) are safety-critical components. Vibration-based fault diagnosis of RTAs is challenging due to the high-dimensional nature of multi-sensor time-series signals and the need for real-time inference in deployed systems.

**TKS-TDM** solves this by learning to identify a tiny set of *temporal key points* — the most fault-informative time positions in the signal — and using only those points for classification. This reduces computational cost by **>99%** compared to dense Transformer baselines, while maintaining state-of-the-art diagnostic accuracy.

### Framework

![TKS-TDM Framework](framework.jpg)

The model consists of three sequential modules:

| Module | Full Name | Role |
|--------|-----------|------|
| **MSFRM** | Multichannel Signal Feature Representation Module | Hierarchical CNN compresses 9-channel raw signals into a rich temporal feature map |
| **KPSM**  | Key Point Selection Module | Iteratively refines N=6 learnable key-point positions via differentiable sampling |
| **CCM**   | Condition Classification Module | Transformer stack with class token classifies 16 fault conditions |

---

## Method

### Full Pipeline

```
Input x ∈ R^(B × 9 × 5120)          (9 channels × 5120 time steps)
        │
        ▼
┌─────────────────────────────┐
│  ① MSFRM                   │   Conv1d → BN → ReLU → MaxPool × 2
│   Hierarchical Feature      │   → BottleneckLayer × 2
│   Representation            │   Output F ∈ R^(B × 288 × 640)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  ② KPSM   (N = 6 iters)    │   Initialise n=8 uniform positions P_1
│   Key Point Selection       │   Each iter:
│                             │     ① Differentiable sampling (linear interp)
│                             │     ② Positional encoding (learnable W_1)
│                             │     ③ Feature fusion + Transformer layer
│                             │     ④ Offset prediction → P_{t+1} = P_t + O_t
│                             │   Output T_N ∈ R^(B × 8 × 288)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  ③ CCM                     │   [T_cls ; T_N] → 8 × TransformerEncoderLayer
│   Condition Classification  │   → LayerNorm → class token → Linear head
│                             │   Output y ∈ R^(B × 16)
└─────────────────────────────┘
```

### Key Features

- **Differentiable Temporal Sampling**: Custom autograd function enables end-to-end gradient flow from classification loss to sampling position updates.
- **Progressive Key-Point Refinement**: KPSM iteratively converges to the most diagnostic time positions without any manual feature engineering.
- **Extreme Efficiency**: Only **8 out of 640** compressed time points are used for classification — fewer than 0.5% of the signal.
- **11.76M parameters, 0.17B FLOPs** — CPU inference latency ~20 ms/sample.

---

## Results

### Railway Test Line Dataset

| Model | ACC (%) | F1 (%) | AUC (%) | Params (M) | FLOPs (B) |
|-------|---------|--------|---------|------------|-----------|
| AlexNet | 91.04 ± 3.11 | 90.91 ± 3.13 | 97.98 ± 1.15 | 57.04 | 3.07 |
| Wide-ResNet | 98.31 ± 0.78 | 98.27 ± 0.82 | 99.75 ± 0.14 | 66.85 | 16.76 |
| BiLSTM | 96.39 ± 1.29 | 96.32 ± 1.34 | 99.34 ± 0.41 | 3.54 | 0.68 |
| MSiT | 97.10 ± 1.47 | 97.06 ± 1.53 | 99.38 ± 0.59 | 24.32 | 2.17 |
| DS-WCNN | 98.06 ± 0.81 | 98.08 ± 0.80 | 99.72 ± 0.17 | 2.10 | 0.41 |
| FDCSANet | 97.77 ± 0.91 | 97.74 ± 0.92 | 99.61 ± 0.21 | 2.77 | 0.54 |
| LD-RPMNet | 97.51 ± 1.34 | 97.50 ± 1.35 | 99.50 ± 0.44 | 12.00 | 0.25 |
| **TKS-TDM (Ours)** | **99.66 ± 0.34** | **99.91 ± 0.09** | **99.87 ± 0.07** | **11.76** | **0.17** |

### Field Deployment Dataset

| Model | ACC (%) | F1 (%) | AUC (%) |
|-------|---------|--------|---------|
| **TKS-TDM (Ours)** | **95.66 ± 1.05** | **95.55 ± 1.12** | **99.38 ± 0.22** |

*All results: mean ± std over 10 independent runs.*

---

## Repository Structure

```
TKS-TDM/
├── models/
│   ├── __init__.py          # Package exports
│   ├── tkstdm.py            # ★ Full TKS-TDM model (MSFRM + KPSM + CCM)
│   ├── kpsm.py              # Key Point Selection Module + DifferentiableSampler
│   └── transformer_block.py # TransformerEncoderLayer, Attention, Mlp (FFN)
│
├── utils/
│   └── __init__.py
│
├── assets/
│   └── framework.jpg        # Method overview figure
│
├── train.py                 # Training script
├── quick_test.py            # Quick sanity-check and evaluation script
├── config.yaml              # Hyperparameter configuration file
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/huang-yu-han/TKS-TDM.git
cd TKS-TDM

# 2. Create a virtual environment (recommended)
conda create -n tkstdm python=3.9
conda activate tkstdm

# 3. Install PyTorch (CUDA 11.7 example; adjust for your CUDA version)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Verify installation (no data needed)

```bash
python quick_test.py
```

Expected output:
```
Device: cuda   (or cpu if no GPU)

Model: TKS-TDM
  Parameters   : 11.76 M
  Embed dim    : 288
  KPSM iters   : 6
  Key points   : 8
  CCM layers   : 8
  Input shape  : 9 × 5119

--- Sanity check (random input, forward pass) ---
  Input  shape : [4, 9, 5119]
  Logits shape : [4, 16]
  Key pts shape: [4, 8, 1]
  Forward pass  ✓
```

### 2. Benchmark CPU latency

```bash
python quick_test.py --benchmark --device cpu
```

### 3. Evaluate a trained checkpoint on your test data

```bash
python quick_test.py \
    --checkpoint checkpoints/best_model.pth \
    --data_path  your/test/data.npy \
    --label_path your/test/labels.npy
```

---

## Training

### Step 1 — Prepare your data

Your data should be provided as NumPy arrays:

| File | Shape | dtype | Description |
|------|-------|-------|-------------|
| `data.npy` | `(N, 9, 5120)` | `float32` | Multi-sensor signals (N samples, 9 channels, 5120 time steps) |
| `labels.npy` | `(N,)` | `int64` | Integer class labels `0 … 15` |

Channel-wise z-score normalisation is applied automatically inside `train.py`.

> **Note:** The dataset used in the paper is proprietary and not publicly released. You can adapt the code to your own multi-sensor vibration dataset by modifying the `load_data()` function in `train.py`.

### Step 2 — Configure `config.yaml`

Update the data paths in `config.yaml`:

```yaml
data:
  train_data_path:  "path/to/your/train_data.npy"
  train_label_path: "path/to/your/train_labels.npy"
  val_data_path:    "path/to/your/val_data.npy"
  val_label_path:   "path/to/your/val_labels.npy"
  test_data_path:   "path/to/your/test_data.npy"
  test_label_path:  "path/to/your/test_labels.npy"
```

All other hyperparameters in `config.yaml` reproduce the paper's settings.

### Step 3 — Run training

```bash
python train.py --config config.yaml
```

The script will:
1. Split data into 64% / 20% / 16% (train / val / test) using stratified sampling.
2. Train for 100 epochs with AdamW + cosine LR schedule + label-smoothing CE.
3. Save the best checkpoint (by validation accuracy) to `checkpoints/best_model.pth`.
4. Report final test ACC / F1 / AUC.

---

## Hyperparameters

All hyperparameters are stored in `config.yaml`. Key settings:

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| `num_points` | n | 8 | Number of temporal key points |
| `embed_dim` | emb | 288 | Feature embedding dimension |
| `num_iters` | N | 6 | KPSM key-point selection iterations |
| `depth` | — | 14 | Total Transformer layers (6 KPSM + 8 CCM) |
| `num_heads` | h | 6 | MSA attention heads |
| `downsample_ratio` | dr | 8 | MSFRM temporal downsampling factor |
| `lr` | — | 5e-4 | Initial learning rate |
| `weight_decay` | — | 0.05 | AdamW weight decay |
| `warmup_epochs` | — | 3 | LR warm-up epochs |
| `label_smoothing` | — | 0.1 | Label smoothing factor |
| `ema_decay` | — | 0.99996 | Model EMA decay rate |

---

## Module Descriptions

### MSFRM — Multichannel Signal Feature Representation Module

Located in `models/tkstdm.py` (`TKSTDM.msfrm`).

Compresses raw 9-channel signals of length L=5120 into a compact feature map F ∈ R^(emb × L/dr):

```
Conv1d(9→64, k=7, s=2) → BN → ReLU
MaxPool1d(k=3, s=2)
MaxPool1d(k=3, s=2)
BottleneckLayer(64 → 64 → 288)    [compress-transform-expand + residual]
BottleneckLayer(288 → 64 → 288)   [compress-transform-expand + residual]
```

### KPSM — Key Point Selection Module

Located in `models/kpsm.py` (`KPSMLayer`).

Iterative module that refines n=8 temporal key-point positions:

- **`DifferentiableSampler`**: Custom autograd 1-D linear interpolation. Enables gradient flow from loss → offset predictions → sampling positions.
- **`KPSMLayer`**: One KPSM iteration (sample → positional encode → fuse → Transformer → predict offset).

### CCM — Condition Classification Module

Located in `models/tkstdm.py` (`TKSTDM.ccm_layers`).

Stack of 8 standard `TransformerEncoderLayer` blocks (from `models/transformer_block.py`) with a prepended learnable class token. The class token output is passed to a linear classification head.

---

## Citing

If you find this work useful, please cite:

```bibtex
@article{huang2026tkstdm,
  title   = {TKS-TDM: A Temporal Key Point Selection Model for Efficient Fault Diagnosis
             of Railway Turnout Actuators Using Multi-Sensor Signals},
  author  = {Huang, Yuhan and Hu, Xiaoxi and He, Yiming and Cao, Jingming and
             Tang, Tao and Shen, Weiming},
  journal = {IEEE Transactions on Transportation Electrification},
  year    = {2026},
  doi     = {10.1109/TTE.2026.3684854}
}
```

---

## License

This project is released for research purposes. The dataset used in the paper is proprietary and is **not** included in this repository.

---

## Acknowledgements

This work was supported by the Guangxi Science and Technology Major Program (Guike AB22035008).

The Transformer building blocks and training utilities build upon [timm](https://github.com/rwightman/pytorch-image-models).
