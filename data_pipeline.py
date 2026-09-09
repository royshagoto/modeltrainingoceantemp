"""
data_pipeline.py
=================
Preprocessing & harmonization pipeline for multi-source satellite / ocean
datasets, and the PyTorch Dataset that feeds the training loop.

Responsibilities
-----------------
1. Regrid every raw source (native resolution/grid) onto the common
   0.25 deg x 0.25 deg North Indian Ocean grid, daily cadence.
2. Harmonize units, fill land / missing values with a sentinel + mask.
3. Normalize each channel (z-score, stats computed on the training period only).
4. Provide a `SubsurfaceTempDataset` (PyTorch) that yields:
       X : (TIME_WINDOW * N_INPUT_CHANNELS, H, W)  surface predictors
       y : (N_DEPTHS, H, W)                        GLORYS temperature target
       m : (N_DEPTHS, H, W)                        valid-data mask (1=ocean/obs)

If `xesmf` is available it is used for conservative/bilinear regridding
(recommended for production use with irregular source grids). Otherwise we
fall back to xarray's native linear interpolation, which is adequate for
already-gridded lat/lon satellite products.
"""

import os
import glob
import json
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset

import config as C

try:
    import xesmf as xe
    HAS_XESMF = True
except ImportError:
    HAS_XESMF = False


# ---------------------------------------------------------------------------
# Regridding
# ---------------------------------------------------------------------------
def build_target_grid():
    """Common analysis grid as an xarray Dataset (for xesmf) and raw arrays."""
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], C.LAT_GRID),
            "lon": (["lon"], C.LON_GRID),
        }
    )
    return ds_out


def regrid_to_common_grid(da: xr.DataArray, method: str = "bilinear") -> xr.DataArray:
    """
    Regrid a single-variable DataArray (dims: time, lat, lon OR time, y, x)
    onto the standardized 0.25 deg North Indian Ocean grid.
    """
    ds_out = build_target_grid()

    if HAS_XESMF:
        regridder = xe.Regridder(da, ds_out, method=method, periodic=False,
                                  ignore_degenerate=True)
        out = regridder(da, keep_attrs=True)
    else:
        # Fallback: linear interpolation along lat/lon (assumes source is
        # already on a regular lat/lon grid, which is true for most
        # gridded SST/SSS/SSH/current/wind satellite products).
        out = da.interp(lat=C.LAT_GRID, lon=C.LON_GRID, method="linear")

    return out


def harmonize_dataset(raw_path: str, varname: str, target_name: str,
                       unit_scale: float = 1.0, unit_offset: float = 0.0) -> xr.DataArray:
    """
    Load one raw NetCDF, select the variable, regrid it onto the common
    grid, apply unit conversion, and return a clean DataArray named
    `target_name` with dims (time, lat, lon).
    """
    ds = xr.open_dataset(raw_path)
    da = ds[varname]

    # standardize coordinate names
    rename_map = {}
    for cand in ["latitude", "Latitude", "y"]:
        if cand in da.coords:
            rename_map[cand] = "lat"
    for cand in ["longitude", "Longitude", "x"]:
        if cand in da.coords:
            rename_map[cand] = "lon"
    if rename_map:
        da = da.rename(rename_map)

    da = regrid_to_common_grid(da)
    da = da * unit_scale + unit_offset
    da = da.resample(time="1D").mean() if "time" in da.dims else da
    da.name = target_name
    return da


def build_daily_stack(source_specs: dict, out_path: str) -> xr.Dataset:
    """
    source_specs: dict of {var_name: dict(path=..., varname=..., scale=1.0, offset=0.0)}
    Harmonizes every source and merges into one daily Dataset written to
    `out_path` (NetCDF) for reuse.
    """
    das = []
    for var_name, spec in source_specs.items():
        da = harmonize_dataset(
            spec["path"], spec["varname"], var_name,
            unit_scale=spec.get("scale", 1.0),
            unit_offset=spec.get("offset", 0.0),
        )
        das.append(da)

    merged = xr.merge(das, join="inner")  # inner join -> common overlapping days
    merged.to_netcdf(out_path)
    return merged


# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------
def compute_and_save_norm_stats(ds: xr.Dataset, var_names, train_time_slice,
                                 out_json: str):
    """Compute per-channel mean/std over the *training* period only and persist."""
    stats = {}
    train_ds = ds.sel(time=train_time_slice)
    for v in var_names:
        arr = train_ds[v].values
        arr = arr[np.isfinite(arr)]
        stats[v] = {"mean": float(np.mean(arr)), "std": float(np.std(arr) + 1e-8)}
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def load_norm_stats(path: str):
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class SubsurfaceTempDataset(Dataset):
    """
    Serves (X, y, mask) samples for the reconstruction model.

    Parameters
    ----------
    surface_ds : xr.Dataset
        Harmonized daily surface predictors on the common grid, with
        variables matching config.INPUT_VARIABLES keys, dims (time, lat, lon).
    target_ds : xr.Dataset
        GLORYS temperature target, variable "thetao" (or "temperature") with
        dims (time, depth, lat, lon), depth already interpolated to
        config.DEPTH_LEVELS.
    norm_stats : dict
        Per-variable {mean, std} computed on the training split.
    time_index : array-like of datetime64
        The subset of timestamps this dataset instance should serve
        (i.e. the train / val / test split boundary is enforced by the
        caller via this argument, so there is no leakage).
    time_window : int
        Number of trailing days of surface fields stacked as extra channels.
    """

    def __init__(self, surface_ds: xr.Dataset, target_ds: xr.Dataset,
                 norm_stats: dict, time_index, time_window: int = C.TIME_WINDOW):
        self.surface_ds = surface_ds
        self.target_ds = target_ds
        self.norm_stats = norm_stats
        self.time_window = time_window
        self.var_names = list(C.INPUT_VARIABLES.keys())

        # Only keep timestamps where we have `time_window` consecutive days
        # of surface data AND a target field.
        all_times = np.sort(np.asarray(surface_ds["time"].values))
        valid_times = []
        target_time_set = set(np.asarray(target_ds["time"].values))

        for t in time_index:
            t = np.datetime64(t)
            if t not in target_time_set:
                continue
            idx = np.searchsorted(all_times, t)
            if idx - (time_window - 1) < 0:
                continue
            window = all_times[idx - time_window + 1: idx + 1]
            if len(window) == time_window and window[-1] == t:
                valid_times.append(t)

        self.times = valid_times
        # target depth name in dataset
        self.target_var = "thetao" if "thetao" in target_ds else list(target_ds.data_vars)[0]

    def __len__(self):
        return len(self.times)

    def _normalize(self, arr, var):
        m, s = self.norm_stats[var]["mean"], self.norm_stats[var]["std"]
        return (arr - m) / s

    def __getitem__(self, idx):
        t = self.times[idx]
        all_times = np.sort(np.asarray(self.surface_ds["time"].values))
        end_idx = np.searchsorted(all_times, t)
        window_times = all_times[end_idx - self.time_window + 1: end_idx + 1]

        # ---- build X: (time_window * n_channels, H, W) ----
        channel_stack = []
        for wt in window_times:
            day_slice = self.surface_ds.sel(time=wt)
            for var in self.var_names:
                arr = day_slice[var].values.astype(np.float32)
                arr = np.nan_to_num(arr, nan=self.norm_stats[var]["mean"])
                arr = self._normalize(arr, var)
                channel_stack.append(arr)
        X = np.stack(channel_stack, axis=0)  # (T*C, H, W)

        # ---- build y and mask: (N_DEPTHS, H, W) ----
        target_slice = self.target_ds[self.target_var].sel(time=t)
        y = target_slice.values.astype(np.float32)          # (depth, H, W)
        mask = np.isfinite(y).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0)

        return (
            torch.from_numpy(X),
            torch.from_numpy(y),
            torch.from_numpy(mask),
            str(np.datetime_as_string(t, unit="D")),
        )


def chronological_split(time_index):
    """Split a sorted array of dates into train/val/test with no shuffling,
    since ocean states are strongly autocorrelated in time."""
    n = len(time_index)
    n_train = int(n * C.TRAIN_FRAC)
    n_val = int(n * C.VAL_FRAC)
    train_times = time_index[:n_train]
    val_times = time_index[n_train:n_train + n_val]
    test_times = time_index[n_train + n_val:]
    return train_times, val_times, test_times
