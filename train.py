"""
train.py
========
Training driver for the Attention-ResUNet subsurface temperature
reconstruction model.

Usage
-----
    python train.py --surface data/interim/surface_daily.nc \
                     --target  data/interim/glorys_temperature.nc

If you don't have real data yet, run `python generate_synthetic_data.py`
first -- it creates a small physically-plausible synthetic dataset in
data/interim/ so you can validate the full pipeline end-to-end before
plugging in real GLORYS / satellite files.
"""

import os
import json
import argparse
import time
import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader

import config as C
from data_pipeline import (SubsurfaceTempDataset, chronological_split,
                            compute_and_save_norm_stats, load_norm_stats)
from model import AttentionResUNet, MaskedDepthWeightedLoss
from utils import set_seed, masked_rmse, masked_correlation, masked_bias, EarlyStopper


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloaders(surface_path, target_path):
    surface_ds = xr.open_dataset(surface_path)
    target_ds = xr.open_dataset(target_path)

    all_times = np.sort(np.asarray(surface_ds["time"].values))
    train_t, val_t, test_t = chronological_split(all_times)

    stats_path = os.path.join(C.PROCESSED_DIR, "norm_stats.json")
    if not os.path.exists(stats_path):
        norm_stats = compute_and_save_norm_stats(
            surface_ds, list(C.INPUT_VARIABLES.keys()),
            train_time_slice=slice(str(train_t[0])[:10], str(train_t[-1])[:10]),
            out_json=stats_path,
        )
    else:
        norm_stats = load_norm_stats(stats_path)

    train_ds = SubsurfaceTempDataset(surface_ds, target_ds, norm_stats, train_t)
    val_ds = SubsurfaceTempDataset(surface_ds, target_ds, norm_stats, val_t)
    test_ds = SubsurfaceTempDataset(surface_ds, target_ds, norm_stats, test_t)

    print(f"Samples -> train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    rmses = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for X, y, mask, _ in loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)

            if train:
                optimizer.zero_grad()

            pred = model(X)
            loss = criterion(pred, y, mask)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            rmses.append(masked_rmse(pred.detach(), y, mask))

    return total_loss / max(n_batches, 1), float(np.mean(rmses))


def train_model(surface_path, target_path, resume=False):
    set_seed(C.SEED)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(surface_path, target_path)

    model = AttentionResUNet().to(device)
    print(f"Model parameters: {model.num_parameters():,}")

    criterion = MaskedDepthWeightedLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LEARNING_RATE,
                                   weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=C.LR_SCHEDULER_FACTOR,
        patience=C.LR_SCHEDULER_PATIENCE
    )

    ckpt_path = os.path.join(C.CHECKPOINT_DIR, "best_model.pt")
    stopper = EarlyStopper(patience=C.EARLY_STOPPING_PATIENCE)

    start_epoch = 0
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    history = {"train_loss": [], "val_loss": [], "train_rmse": [], "val_rmse": []}
    log_path = os.path.join(C.LOG_DIR, "training_log.json")

    for epoch in range(start_epoch, C.EPOCHS):
        t0 = time.time()
        train_loss, train_rmse = run_epoch(model, train_loader, criterion, optimizer,
                                            device, train=True)
        val_loss, val_rmse = run_epoch(model, val_loader, criterion, optimizer,
                                        device, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:03d}/{C.EPOCHS} | "
              f"train_loss {train_loss:.4f} rmse {train_rmse:.3f}C | "
              f"val_loss {val_loss:.4f} rmse {val_rmse:.3f}C | "
              f"lr {lr_now:.2e} | {dt:.1f}s")

        improved = stopper.step(val_loss)
        if improved:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": {k: v for k, v in vars(C).items() if k.isupper()},
            }, ckpt_path)
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.4f})")

        if stopper.should_stop:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    print("\nTraining complete. Evaluating best checkpoint on held-out test split...")
    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    test_loss, test_rmse = run_epoch(model, test_loader, criterion, optimizer,
                                      device, train=False)
    print(f"TEST  | loss {test_loss:.4f} | rmse {test_rmse:.3f} C")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", default=os.path.join(C.INTERIM_DIR, "surface_daily.nc"))
    parser.add_argument("--target", default=os.path.join(C.INTERIM_DIR, "glorys_temperature.nc"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train_model(args.surface, args.target, resume=args.resume)
