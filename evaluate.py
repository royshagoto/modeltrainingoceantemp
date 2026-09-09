"""
evaluate.py
===========
Validates the trained reconstruction model against INDEPENDENT gridded
ARGO observations (not used in training), computing correlation, RMSE,
and bias at every standard depth level -- as required by the problem
statement's "Evaluate the reconstruction using independent observations
and standard skill metrics" requirement.

Usage
-----
    python evaluate.py --surface data/interim/surface_daily.nc \
                        --target  data/interim/glorys_temperature.nc \
                        --argo    data/raw/gridded_argo_temperature.nc
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import xarray as xr
import torch

import config as C
from data_pipeline import SubsurfaceTempDataset, chronological_split, load_norm_stats
from model import AttentionResUNet
from utils import per_depth_metrics, masked_rmse, masked_correlation, masked_bias


def load_model(ckpt_path, device):
    model = AttentionResUNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    return model


@torch.no_grad()
def reconstruct_all(model, dataset, device, batch_size=8):
    """Run the model over an entire dataset split, returning stacked
    (pred, target, mask, dates) for later ARGO comparison / plotting."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds, targets, masks, dates = [], [], [], []
    for X, y, m, d in loader:
        X = X.to(device)
        pred = model(X).cpu().numpy()
        preds.append(pred)
        targets.append(y.numpy())
        masks.append(m.numpy())
        dates.extend(d)

    return (np.concatenate(preds, axis=0),
            np.concatenate(targets, axis=0),
            np.concatenate(masks, axis=0),
            dates)


def evaluate_against_glorys(model, surface_path, target_path, device, split="test"):
    """Skill metrics vs. the GLORYS reanalysis target used for training
    (sanity check of reconstruction fidelity on the model's own target)."""
    surface_ds = xr.open_dataset(surface_path)
    target_ds = xr.open_dataset(target_path)
    norm_stats = load_norm_stats(os.path.join(C.PROCESSED_DIR, "norm_stats.json"))

    all_times = np.sort(np.asarray(surface_ds["time"].values))
    train_t, val_t, test_t = chronological_split(all_times)
    times = {"train": train_t, "val": val_t, "test": test_t}[split]

    ds = SubsurfaceTempDataset(surface_ds, target_ds, norm_stats, times)
    pred, target, mask, dates = reconstruct_all(model, ds, device)

    metrics = per_depth_metrics(pred, target, mask, C.DEPTH_LEVELS)
    df = pd.DataFrame(metrics)
    print(f"\n=== Skill vs. GLORYS reanalysis target ({split} split, n={len(dates)} days) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df, pred, target, mask, dates


def evaluate_against_argo(model, surface_path, argo_path, device, time_window=C.TIME_WINDOW):
    """
    Independent validation: compare model reconstructions to Gridded ARGO
    profiles that were NOT used anywhere in training (neither as input nor
    as the GLORYS training target). This is the key "independent
    observations" skill check requested in the problem statement.
    """
    surface_ds = xr.open_dataset(surface_path)
    argo_ds = xr.open_dataset(argo_path)
    norm_stats = load_norm_stats(os.path.join(C.PROCESSED_DIR, "norm_stats.json"))

    argo_var = "temperature" if "temperature" in argo_ds else list(argo_ds.data_vars)[0]
    # interpolate ARGO onto the same standard depth levels if needed
    if "depth" in argo_ds.dims and list(argo_ds["depth"].values) != C.DEPTH_LEVELS:
        argo_ds = argo_ds.interp(depth=C.DEPTH_LEVELS)

    common_times = np.intersect1d(
        np.asarray(surface_ds["time"].values), np.asarray(argo_ds["time"].values)
    )
    if len(common_times) == 0:
        print("No overlapping dates between surface predictors and ARGO obs. Skipping.")
        return None

    # Build a lightweight "target_ds" wrapper matching ARGO so we can reuse
    # the SubsurfaceTempDataset machinery unchanged.
    argo_target_ds = argo_ds.rename({argo_var: "thetao"}) if argo_var != "thetao" else argo_ds

    ds = SubsurfaceTempDataset(surface_ds, argo_target_ds, norm_stats, common_times,
                                time_window=time_window)
    if len(ds) == 0:
        print("No valid overlapping samples with sufficient time_window history. Skipping.")
        return None

    pred, target, mask, dates = reconstruct_all(model, ds, device)
    metrics = per_depth_metrics(pred, target, mask, C.DEPTH_LEVELS)
    df = pd.DataFrame(metrics)
    print(f"\n=== INDEPENDENT skill vs. Gridded ARGO (n={len(dates)} days) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    out_csv = os.path.join(C.OUTPUT_DIR, "argo_independent_skill_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", default=os.path.join(C.INTERIM_DIR, "surface_daily.nc"))
    parser.add_argument("--target", default=os.path.join(C.INTERIM_DIR, "glorys_temperature.nc"))
    parser.add_argument("--argo", default=C.ARGO_NC_PATH)
    parser.add_argument("--checkpoint", default=os.path.join(C.CHECKPOINT_DIR, "best_model.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    glorys_df, pred, target, mask, dates = evaluate_against_glorys(
        model, args.surface, args.target, device, split="test"
    )
    glorys_df.to_csv(os.path.join(C.OUTPUT_DIR, "glorys_test_skill_metrics.csv"), index=False)

    if os.path.exists(args.argo):
        evaluate_against_argo(model, args.surface, args.argo, device)
    else:
        print(f"\nARGO file not found at {args.argo} -- skipping independent validation. "
              f"Point --argo at your INCOIS LAS Gridded ARGO NetCDF to run it.")
