"""
quick_test.py — Quick sanity-check and model evaluation script for TKS-TDM.

This script verifies the TKS-TDM model with synthetic data (no real dataset
required) and, optionally, evaluates a trained checkpoint on real test data.

Usage
-----
# 1. Sanity check with random data (no checkpoint needed):
    python quick_test.py

# 2. Evaluate a checkpoint on your test data:
    python quick_test.py --checkpoint checkpoints/best_model.pth \\
                         --data_path  your/test/data.npy        \\
                         --label_path your/test/labels.npy

# 3. Also measure CPU inference latency:
    python quick_test.py --benchmark
"""

import argparse
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from models import TKSTDM


# ---------------------------------------------------------------------------
# Default model configuration (matches paper / config.yaml)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_CFG = dict(
    signal_length    = 5119,
    num_points       = 8,
    in_channels      = 9,
    downsample_ratio = 8,
    num_classes      = 16,
    num_iters        = 6,
    depth            = 14,
    embed_dim        = 288,
    num_heads        = 6,
    mlp_ratio        = 3.0,
    offset_gamma     = 1.0,
    offset_bias      = True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def channel_wise_normalize(data: np.ndarray) -> np.ndarray:
    """Per-channel z-score normalisation.  Shape: (N, C, L)."""
    out = np.zeros_like(data)
    for i in range(data.shape[0]):
        for c in range(data.shape[1]):
            sig = data[i, c]
            out[i, c] = (sig - sig.mean()) / (sig.std() + 1e-8)
    return out


def build_model(cfg: dict, device: torch.device) -> TKSTDM:
    model = TKSTDM(**cfg).to(device)
    return model


def count_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def run_evaluation(model, data: np.ndarray, labels: np.ndarray,
                   batch_size: int, device: torch.device, num_classes: int):
    model.eval()
    x_t = torch.from_numpy(data).float()
    y_t = torch.from_numpy(labels).long()
    dataset  = torch.utils.data.TensorDataset(x_t, y_t)
    loader   = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False)

    all_preds, all_labels, all_probs = [], [], []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        logits, _ = model(x_batch)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y_batch.numpy())
        all_probs.append(probs)

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro')
    y_bin = label_binarize(all_labels, classes=range(num_classes))
    try:
        auc = roc_auc_score(y_bin, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')

    return acc, f1, auc


def measure_latency(model, device, signal_length: int, in_channels: int,
                    batch_sizes=(1, 8, 16, 32), n_warmup=20, n_runs=100):
    """Measure per-sample CPU/GPU inference latency and throughput."""
    model.eval()
    results = {}
    for bs in batch_sizes:
        dummy = torch.randn(bs, in_channels, signal_length).to(device)
        with torch.no_grad():
            for _ in range(n_warmup):
                model(dummy)
        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model(dummy)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)  # ms
        lat = np.array(latencies)
        results[bs] = dict(mean=lat.mean(), std=lat.std(),
                           p50=np.percentile(lat, 50),
                           p95=np.percentile(lat, 95),
                           throughput=bs / (lat.mean() / 1000))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='TKS-TDM Quick Test')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to a trained checkpoint (.pth)')
    parser.add_argument('--data_path',  type=str, default='',
                        help='Path to test data  (.npy, shape N×C×L)')
    parser.add_argument('--label_path', type=str, default='',
                        help='Path to test labels (.npy, shape N)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_classes',type=int, default=16)
    parser.add_argument('--benchmark', action='store_true',
                        help='Run latency benchmark after model check')
    parser.add_argument('--device',    type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Device: {device}')

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    cfg = {**DEFAULT_MODEL_CFG, 'num_classes': args.num_classes}
    model = build_model(cfg, device)
    n_params = count_params(model)
    print(f'\nModel: TKS-TDM')
    print(f'  Parameters   : {n_params:.2f} M')
    print(f'  Embed dim    : {cfg["embed_dim"]}')
    print(f'  KPSM iters   : {cfg["num_iters"]}')
    print(f'  Key points   : {cfg["num_points"]}')
    print(f'  CCM layers   : {cfg["depth"] - cfg["num_iters"]}')
    print(f'  Input shape  : {cfg["in_channels"]} × {cfg["signal_length"]}')

    # -----------------------------------------------------------------------
    # Load checkpoint (optional)
    # -----------------------------------------------------------------------
    if args.checkpoint:
        import os
        if not os.path.isfile(args.checkpoint):
            print(f'[WARNING] Checkpoint not found: {args.checkpoint}')
        else:
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            best_acc = ckpt.get('best_acc', float('nan'))
            epoch    = ckpt.get('epoch', '?')
            print(f'\nLoaded checkpoint: {args.checkpoint}')
            print(f'  Saved epoch : {epoch}')
            print(f'  Best val acc: {best_acc:.4f}')

    # -----------------------------------------------------------------------
    # Sanity check with random synthetic data
    # -----------------------------------------------------------------------
    print('\n--- Sanity check (random input, forward pass) ---')
    model.eval()
    dummy = torch.randn(4, cfg['in_channels'], cfg['signal_length']).to(device)
    with torch.no_grad():
        logits, key_pts = model(dummy)
    print(f'  Input  shape : {list(dummy.shape)}')
    print(f'  Logits shape : {list(logits.shape)}')
    print(f'  Key pts shape: {list(key_pts.shape)}')
    print('  Forward pass  ✓')

    # -----------------------------------------------------------------------
    # Evaluate on real test data (if provided)
    # -----------------------------------------------------------------------
    if args.data_path and args.label_path:
        import os
        if not os.path.isfile(args.data_path):
            print(f'[ERROR] Data file not found: {args.data_path}')
        elif not os.path.isfile(args.label_path):
            print(f'[ERROR] Label file not found: {args.label_path}')
        else:
            print(f'\n--- Evaluating on test data ---')
            data   = np.load(args.data_path).astype(np.float32)
            labels = np.load(args.label_path).astype(np.int64)
            data   = channel_wise_normalize(data)
            print(f'  Test samples : {len(labels)}')
            print(f'  Data shape   : {data.shape}')

            acc, f1, auc = run_evaluation(model, data, labels,
                                           args.batch_size, device,
                                           args.num_classes)
            print(f'\n  ACC  : {acc * 100:.2f}%')
            print(f'  F1   : {f1 * 100:.2f}%')
            print(f'  AUC  : {auc * 100:.2f}%')
    else:
        print('\n[INFO] No test data provided. '
              'Pass --data_path and --label_path to evaluate on real data.')

    # -----------------------------------------------------------------------
    # Latency benchmark (optional)
    # -----------------------------------------------------------------------
    if args.benchmark:
        bench_device = torch.device('cpu')
        model_cpu = model.to(bench_device)
        print(f'\n--- Latency Benchmark (CPU) ---')
        results = measure_latency(model_cpu, bench_device,
                                  cfg['signal_length'], cfg['in_channels'])
        print(f'  {"Batch":>6}  {"Mean (ms)":>10}  {"P95 (ms)":>10}  '
              f'{"Throughput":>14}')
        for bs, r in results.items():
            print(f'  {bs:>6}  {r["mean"]:>10.2f}  {r["p95"]:>10.2f}  '
                  f'{r["throughput"]:>10.1f} sps')
        print()


if __name__ == '__main__':
    main()
