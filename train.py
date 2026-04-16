"""
train.py — Training script for TKS-TDM.

Usage:
    python train.py --config config.yaml

The script reads all hyperparameters from config.yaml (or command-line
overrides) and performs:
  1. Data loading (from user-supplied .npy files).
  2. Model construction (TKSTDM).
  3. Training with AdamW + cosine LR schedule + label-smoothing CE loss.
  4. EMA model tracking.
  5. Validation after each epoch; best checkpoint saved automatically.
  6. Final evaluation on the test set.

NOTE: Provide your own data loading logic in the `load_data()` function
below. The expected data format is NumPy arrays:
    data   : float32, shape (N, C, L)   — N samples, C channels, L time steps
    labels : int64,   shape (N,)        — integer class indices 0 … num_classes-1
"""

import argparse
import os
import logging
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

import yaml
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import AverageMeter

from models import TKSTDM


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='TKS-TDM Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML configuration file')
    # Allow command-line overrides for key training params
    parser.add_argument('--batch_size',    type=int,   default=None)
    parser.add_argument('--epochs',        type=int,   default=None)
    parser.add_argument('--lr',            type=float, default=None)
    parser.add_argument('--num_classes',   type=int,   default=None)
    parser.add_argument('--amp',           action='store_true', default=None)
    parser.add_argument('--resume',        type=str,   default=None,
                        help='Path to checkpoint to resume from')
    # timm scheduler / optimizer compat flags
    parser.add_argument('--opt',           type=str,   default='adamw')
    parser.add_argument('--weight-decay',  type=float, default=None)
    parser.add_argument('--momentum',      type=float, default=0.9)
    parser.add_argument('--clip-grad',     type=float, default=None)
    parser.add_argument('--sched',         type=str,   default='cosine')
    parser.add_argument('--warmup-epochs', type=int,   default=None)
    parser.add_argument('--min-lr',        type=float, default=None)
    parser.add_argument('--cooldown-epochs', type=int, default=10)
    parser.add_argument('--drop',          type=float, default=0.0)
    parser.add_argument('--drop-path',     type=float, default=0.1)
    parser.add_argument('--smoothing',     type=float, default=None)
    parser.add_argument('--model-ema',     action='store_true', default=True)
    parser.add_argument('--model-ema-decay', type=float, default=None)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_args(args, cfg: dict):
    """Fill argparse Namespace from YAML config where CLI arg is None."""
    tr = cfg.get('training', {})
    sc = cfg.get('scheduler', {})
    em = cfg.get('ema', {})

    if args.batch_size    is None: args.batch_size    = tr.get('batch_size', 32)
    if args.epochs        is None: args.epochs        = tr.get('epochs', 100)
    if args.lr            is None: args.lr            = tr.get('lr', 5e-4)
    if args.num_classes   is None: args.num_classes   = cfg.get('model', {}).get('num_classes', 16)
    if args.amp           is None: args.amp           = tr.get('amp', True)
    if args.weight_decay  is None: args.weight_decay  = tr.get('weight_decay', 0.05)
    if args.clip_grad     is None: args.clip_grad     = tr.get('clip_grad', 1.0)
    if args.smoothing     is None: args.smoothing     = tr.get('label_smoothing', 0.1)
    if args.warmup_epochs is None: args.warmup_epochs = sc.get('warmup_epochs', 3)
    if args.min_lr        is None: args.min_lr        = sc.get('min_lr', 1e-5)
    if args.model_ema_decay is None: args.model_ema_decay = em.get('decay', 0.99996)
    return args


# ---------------------------------------------------------------------------
# Data loading  ← Implement your own dataset here
# ---------------------------------------------------------------------------

def channel_wise_normalize(data: np.ndarray) -> np.ndarray:
    """Z-score normalisation applied independently to each channel.

    Args:
        data: float32 array of shape (N, C, L).

    Returns:
        Normalised array of the same shape.
    """
    out = np.zeros_like(data)
    for i in range(data.shape[0]):
        for c in range(data.shape[1]):
            sig = data[i, c]
            mu, sigma = sig.mean(), sig.std()
            out[i, c] = (sig - mu) / (sigma + 1e-8)
    return out


def load_data(cfg: dict):
    """Load and split data into train / val / test DataLoaders.

    Implement this function according to your data format.
    Expected format:
        data   : (N, C, L) float32 NumPy array
        labels : (N,)      int64   NumPy array

    Returns:
        train_loader, val_loader, test_loader
    """
    data_cfg = cfg.get('data', {})

    # -----------------------------------------------------------------
    # ↓↓↓  Replace the lines below with your own data loading logic  ↓↓↓
    # -----------------------------------------------------------------
    data_path  = data_cfg.get('train_data_path',  'your/data/path')
    label_path = data_cfg.get('train_label_path', 'your/label/path')

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Please update the data paths in config.yaml before training.")

    data   = np.load(data_path).astype(np.float32)   # (N, C, L)
    labels = np.load(label_path).astype(np.int64)     # (N,)
    # -----------------------------------------------------------------
    # ↑↑↑                    End of data loading                    ↑↑↑
    # -----------------------------------------------------------------

    # Channel-wise z-score normalisation
    data = channel_wise_normalize(data)

    # Stratified split: 64% train / 20% val / 16% test
    tr_r = data_cfg.get('train_ratio', 0.64)
    va_r = data_cfg.get('val_ratio',   0.20)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        data, labels, test_size=1 - tr_r, stratify=labels, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp,
        test_size=1 - va_r / (1 - tr_r),
        stratify=y_tmp, random_state=42)

    batch_size  = data_cfg.get('batch_size', cfg.get('training', {}).get('batch_size', 32))
    num_workers = data_cfg.get('num_workers', 4)

    def make_loader(X, y, shuffle=False):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          pin_memory=True)

    return (make_loader(X_tr, y_tr, shuffle=True),
            make_loader(X_va, y_va),
            make_loader(X_te, y_te))


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class ModelEMA(nn.Module):
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.99996):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def update(self, model: nn.Module):
        with torch.no_grad():
            for ema_p, m_p in zip(self.module.parameters(), model.parameters()):
                ema_p.copy_(self.decay * ema_p + (1. - self.decay) * m_p)
            for ema_b, m_b in zip(self.module.buffers(), model.buffers()):
                ema_b.copy_(m_b)


# ---------------------------------------------------------------------------
# Train / Evaluate one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, epoch,
                    scheduler, ema_model, scaler, args):
    model.train()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.float().cuda(), y.long().cuda()

        with torch.cuda.amp.autocast(enabled=args.amp):
            logits, _ = model(x)
            loss = criterion(logits, y)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        if ema_model is not None:
            ema_model.update(model)

        scheduler.step_update(epoch * len(loader) + batch_idx)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = preds.eq(y).float().mean().item()
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc,          x.size(0))

        if batch_idx % args.log_interval == 0:
            logging.info(
                f'Epoch {epoch:03d} [{batch_idx}/{len(loader)}] '
                f'Loss: {loss_meter.avg:.4f}  Acc: {acc_meter.avg:.4f}')

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, num_classes: int = 16):
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels, all_probs = [], [], []

    for x, y in loader:
        x, y = x.float().cuda(), y.long().cuda()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss_meter.update(loss.item(), x.size(0))
        probs = torch.softmax(logits, dim=1)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)

    acc = (all_preds == all_labels).mean()
    f1  = f1_score(all_labels, all_preds, average='macro')

    y_bin = label_binarize(all_labels, classes=range(num_classes))
    try:
        auc = roc_auc_score(y_bin, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')

    return loss_meter.avg, acc, f1, auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    args = merge_args(args, cfg)
    args.log_interval = cfg.get('logging', {}).get('log_interval', 10)

    # Logging setup
    os.makedirs('logs', exist_ok=True)
    log_file = cfg.get('logging', {}).get('log_file', 'logs/training_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    logging.info('=== TKS-TDM Training ===')
    logging.info(f'Config: {args.config}')

    # Data
    train_loader, val_loader, test_loader = load_data(cfg)
    logging.info(f'Splits — train: {len(train_loader.dataset)}, '
                 f'val: {len(val_loader.dataset)}, '
                 f'test: {len(test_loader.dataset)}')

    # Model (TKSTDM)
    mc = cfg.get('model', {})
    model = TKSTDM(
        signal_length   = mc.get('signal_length',   5119),
        num_points      = mc.get('num_points',      8),
        in_channels     = mc.get('in_channels',     9),
        downsample_ratio= mc.get('downsample_ratio',8),
        num_classes     = mc.get('num_classes',     16),
        num_iters       = mc.get('num_iters',       6),
        depth           = mc.get('depth',           14),
        embed_dim       = mc.get('embed_dim',       288),
        num_heads       = mc.get('num_heads',       6),
        mlp_ratio       = mc.get('mlp_ratio',       3.0),
        offset_gamma    = mc.get('offset_gamma',    1.0),
        offset_bias     = mc.get('offset_bias',     True),
        drop_rate       = mc.get('drop_rate',       0.0),
        attn_drop_rate  = mc.get('attn_drop_rate',  0.0),
        drop_path_rate  = mc.get('drop_path_rate',  0.1),
    ).cuda()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f'Parameters: {n_params:.2f}M')

    # EMA
    ema_model = None
    if args.model_ema:
        ema_model = ModelEMA(model, decay=args.model_ema_decay)

    # Optimiser & scheduler
    optimizer          = create_optimizer(args, model)
    scheduler, n_epochs = create_scheduler(args, optimizer)

    # Loss
    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Resume
    start_epoch = 0
    best_acc    = 0.0
    ckpt_path   = cfg.get('checkpoint', {}).get('save_path', 'checkpoints/best_model.pth')
    os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)

    resume = args.resume or cfg.get('checkpoint', {}).get('resume', '')
    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location='cuda')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_acc    = ckpt['best_acc']
        logging.info(f'Resumed from {resume} (epoch {start_epoch}, best_acc {best_acc:.4f})')

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            epoch, scheduler, ema_model, scaler, args)

        eval_model = ema_model.module if ema_model is not None else model
        val_loss, val_acc, val_f1, val_auc = evaluate(
            eval_model, val_loader, criterion, args.num_classes)

        scheduler.step(epoch + 1, val_loss)

        logging.info(
            f'Epoch {epoch+1:03d}/{args.epochs} | '
            f'Train Loss {train_loss:.4f} Acc {train_acc:.4f} | '
            f'Val Loss {val_loss:.4f} Acc {val_acc:.4f} '
            f'F1 {val_f1:.4f} AUC {val_auc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch':              epoch,
                'model_state_dict':   model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc':           best_acc,
            }, ckpt_path)
            logging.info(f'  ✓ Saved best model → {ckpt_path} (acc={best_acc:.4f})')

    # Final test evaluation
    logging.info('=== Final Test Evaluation ===')
    ckpt = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(ckpt['model_state_dict'])
    eval_model = ema_model.module if ema_model is not None else model
    _, test_acc, test_f1, test_auc = evaluate(
        eval_model, test_loader, criterion, args.num_classes)
    logging.info(f'Test  Acc: {test_acc:.4f}  F1: {test_f1:.4f}  AUC: {test_auc:.4f}')


if __name__ == '__main__':
    main()
