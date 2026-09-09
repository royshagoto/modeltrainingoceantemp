"""
utils.py
========
Metrics (RMSE, correlation, bias -- all mask-aware) and small helpers
shared across training / evaluation.
"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Masked metrics (ignore land / missing-observation pixels)
# ---------------------------------------------------------------------------
def masked_rmse(pred, target, mask):
    diff2 = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return torch.sqrt(diff2.sum() / denom).item()


def masked_bias(pred, target, mask):
    diff = (pred - target) * mask
    denom = mask.sum().clamp_min(1.0)
    return (diff.sum() / denom).item()


def masked_mae(pred, target, mask):
    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return (diff.sum() / denom).item()


def masked_correlation(pred, target, mask):
    """
    Pearson correlation over all masked (valid) elements, flattened.
    Works on numpy arrays or tensors; returns a python float.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    m = mask.astype(bool).ravel()
    p = pred.ravel()[m]
    t = target.ravel()[m]
    if len(p) < 2 or np.std(p) < 1e-8 or np.std(t) < 1e-8:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def per_depth_metrics(pred, target, mask, depth_levels):
    """
    pred, target, mask: (N_DEPTHS, H, W) numpy arrays (single sample) or
    (B, N_DEPTHS, H, W) batched -- averaged over batch dim if present.
    Returns a list of dicts, one per depth level.
    """
    if pred.ndim == 4:
        # flatten batch into spatial dim for per-depth pooled stats
        pred = np.moveaxis(pred, 1, 0).reshape(pred.shape[1], -1, pred.shape[2], pred.shape[3])
        pred = pred.reshape(pred.shape[0], -1)
        target = np.moveaxis(target, 1, 0).reshape(target.shape[1], -1)
        mask = np.moveaxis(mask, 1, 0).reshape(mask.shape[1], -1)
    else:
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        mask = mask.reshape(mask.shape[0], -1)

    results = []
    for i, d in enumerate(depth_levels):
        m = mask[i].astype(bool)
        if m.sum() < 2:
            results.append({"depth_m": d, "rmse": float("nan"), "bias": float("nan"),
                             "corr": float("nan"), "n_obs": int(m.sum())})
            continue
        p, t = pred[i][m], target[i][m]
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        bias = float(np.mean(p - t))
        corr = float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 1e-8 and np.std(t) > 1e-8 else float("nan")
        results.append({"depth_m": d, "rmse": rmse, "bias": bias, "corr": corr,
                         "n_obs": int(m.sum())})
    return results


class EarlyStopper:
    """Tracks validation loss and signals when to stop / when a new best was found."""

    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        improved = val_loss < (self.best - self.min_delta)
        if improved:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved
