"""
generate_synthetic_data.py
===========================
Generates a small, PHYSICALLY-PLAUSIBLE synthetic dataset (surface fields
+ subsurface temperature target + a held-out "ARGO-like" independent set)
on the exact North Indian Ocean 0.25 deg grid, so the full pipeline
(data_pipeline -> model -> train -> evaluate) can be run and validated
end-to-end before you plug in real GLORYS / satellite / ARGO files.

The synthetic ocean:
  * has a warm, shallow surface signal that varies with a seasonal cycle
    and propagating mesoscale eddies (via superposed 2D Gaussians drifting
    westward, mimicking real N. Indian Ocean eddy propagation),
  * a target subsurface temperature field built from a smooth vertical
    profile (warm surface -> cold deep) MODULATED by SSH-anomaly-driven
    thermocline displacement (positive SSH anomaly = downwelling = deeper/
    warmer thermocline, negative = upwelling = shallower/colder), which is
    the actual physical mechanism the real model is meant to learn.

This is for pipeline validation / demonstration only -- swap in real
CMEMS/GLORYS/OSTIA/AVISO/CCMP/INCOIS files for a real PoC.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr

import config as C


def make_eddy_field(lat2d, lon2d, t_day, n_eddies=6, seed=0):
    """Superposition of drifting Gaussian eddies -> SSH-like anomaly field (m)."""
    rng = np.random.default_rng(seed)
    field = np.zeros_like(lat2d)
    for k in range(n_eddies):
        amp = rng.uniform(-0.25, 0.25)
        lat0 = rng.uniform(C.LAT_MIN + 3, C.LAT_MAX - 3)
        lon0_start = rng.uniform(C.LON_MIN + 5, C.LON_MAX - 5)
        drift_speed = rng.uniform(-0.05, -0.01)  # deg/day, westward propagation
        lon0 = lon0_start + drift_speed * t_day
        sigma = rng.uniform(2.0, 4.5)
        field += amp * np.exp(-(((lat2d - lat0) ** 2 + (lon2d - lon0) ** 2) / (2 * sigma ** 2)))
    return field


def generate(n_days=730, out_dir=C.INTERIM_DIR, argo_holdout_frac=0.08, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    lat = C.LAT_GRID
    lon = C.LON_GRID
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")  # (H, W)

    dates = pd.date_range("2019-01-01", periods=n_days, freq="D")

    sst = np.zeros((n_days, C.H, C.W), dtype=np.float32)
    sss = np.zeros((n_days, C.H, C.W), dtype=np.float32)
    ssh = np.zeros((n_days, C.H, C.W), dtype=np.float32)
    u_curr = np.zeros((n_days, C.H, C.W), dtype=np.float32)
    v_curr = np.zeros((n_days, C.H, C.W), dtype=np.float32)
    u_wind = np.zeros((n_days, C.H, C.W), dtype=np.float32)
    v_wind = np.zeros((n_days, C.H, C.W), dtype=np.float32)
    temp = np.zeros((n_days, C.N_DEPTHS, C.H, C.W), dtype=np.float32)

    depths = np.array(C.DEPTH_LEVELS, dtype=np.float32)

    for i, d in enumerate(dates):
        doy = d.dayofyear
        seasonal = 2.0 * np.sin(2 * np.pi * (doy - 60) / 365.0)  # warm pre-monsoon peak

        eddy = make_eddy_field(lat2d, lon2d, i, n_eddies=6, seed=seed + i // 30)
        eddy_field = eddy.T  # align shape with (H, W) meshgrid indexing "ij" -> already H,W

        noise = rng.normal(0, 0.15, size=(C.H, C.W)).astype(np.float32)

        # SST: warmer at low latitude, seasonal cycle, eddy SST signature, noise
        base_sst = 29.5 - 0.15 * (lat2d - C.LAT_MIN)
        sst[i] = (base_sst + seasonal + 1.5 * eddy + noise).astype(np.float32)

        # SSS: fresher near Bay of Bengal (higher longitude side, river runoff), saltier Arabian Sea
        sss[i] = (35.5 - 2.0 * np.exp(-((lon2d - 90) ** 2) / (2 * 10 ** 2))
                   + 0.3 * rng.normal(0, 1, size=(C.H, C.W))).astype(np.float32)

        # SSH anomaly directly driven by the eddy field (this IS the eddy field, by construction)
        ssh[i] = eddy.astype(np.float32)

        # Geostrophic-like surface currents ~ derivative of SSH (finite diff)
        dssh_dy, dssh_dx = np.gradient(eddy, lat[1] - lat[0], lon[1] - lon[0])
        f_cor = 2 * 7.29e-5 * np.sin(np.deg2rad(lat2d))
        f_cor = np.where(np.abs(f_cor) < 1e-6, 1e-6, f_cor)
        g = 9.81
        u_curr[i] = (-g / f_cor * dssh_dy * 1e-3).astype(np.float32)
        v_curr[i] = (g / f_cor * dssh_dx * 1e-3).astype(np.float32)

        # Winds: simplified monsoon proxy (SW monsoon Jun-Sep, NE monsoon Dec-Feb)
        month = d.month
        if 6 <= month <= 9:
            u_wind[i] = (5.0 + rng.normal(0, 1, (C.H, C.W))).astype(np.float32)
            v_wind[i] = (3.0 + rng.normal(0, 1, (C.H, C.W))).astype(np.float32)
        elif month in (12, 1, 2):
            u_wind[i] = (-3.0 + rng.normal(0, 1, (C.H, C.W))).astype(np.float32)
            v_wind[i] = (-2.0 + rng.normal(0, 1, (C.H, C.W))).astype(np.float32)
        else:
            u_wind[i] = rng.normal(0, 1.5, (C.H, C.W)).astype(np.float32)
            v_wind[i] = rng.normal(0, 1.5, (C.H, C.W)).astype(np.float32)

        # ---- Subsurface temperature target ----
        # Base climatological vertical profile: warm mixed layer -> sharp
        # thermocline -> cold deep water.
        base_profile = 4.0 + 24.0 * np.exp(-depths / 120.0)   # (N_DEPTHS,)

        # Thermocline displacement driven by SSH anomaly: positive SSH ->
        # downwelling -> deeper/warmer thermocline; negative -> shallower/colder.
        displacement = 40.0 * eddy  # meters of vertical displacement, per grid cell (H, W)

        for k, dz in enumerate(depths):
            # shift the effective depth used to sample the base profile
            eff_depth = np.clip(dz - displacement, 0, 1500)
            level_temp = 4.0 + 24.0 * np.exp(-eff_depth / 120.0)
            level_temp += 0.3 * seasonal * np.exp(-dz / 80.0)  # seasonal signal decays with depth
            level_temp += rng.normal(0, 0.1, size=(C.H, C.W))
            temp[i, k] = level_temp.astype(np.float32)

    # Land mask: simple rectangular "coastline" carve-outs to mimic India's
    # coastline blocking part of the domain (purely illustrative).
    land_mask = np.ones((C.H, C.W), dtype=bool)
    # crude landmass around west coast of India / Sri Lanka gap, illustrative only
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            if 8 <= la <= 23 and 68 <= lo <= 77 and (lo - 68) < (la - 8) * 0.6:
                land_mask[i, j] = False
    for var in [sst, sss, ssh, u_curr, v_curr]:
        var[:, ~land_mask] = np.nan
    temp[:, :, ~land_mask] = np.nan

    surface_ds = xr.Dataset(
        {
            "sst": (["time", "lat", "lon"], sst),
            "sss": (["time", "lat", "lon"], sss),
            "ssh": (["time", "lat", "lon"], ssh),
            "u_curr": (["time", "lat", "lon"], u_curr),
            "v_curr": (["time", "lat", "lon"], v_curr),
            "u_wind": (["time", "lat", "lon"], u_wind),
            "v_wind": (["time", "lat", "lon"], v_wind),
        },
        coords={"time": dates, "lat": lat, "lon": lon},
    )

    target_ds = xr.Dataset(
        {"thetao": (["time", "depth", "lat", "lon"], temp)},
        coords={"time": dates, "depth": depths, "lat": lat, "lon": lon},
    )

    surface_path = os.path.join(out_dir, "surface_daily.nc")
    target_path = os.path.join(out_dir, "glorys_temperature.nc")
    surface_ds.to_netcdf(surface_path)
    target_ds.to_netcdf(target_path)
    print(f"Wrote synthetic surface predictors -> {surface_path}  ({surface_ds.nbytes/1e6:.1f} MB)")
    print(f"Wrote synthetic GLORYS-like target  -> {target_path}  ({target_ds.nbytes/1e6:.1f} MB)")

    # ---- Independent "ARGO-like" holdout: sparse-in-time, add extra obs noise ----
    n_argo_days = max(int(n_days * argo_holdout_frac), 10)
    argo_idx = np.sort(rng.choice(n_days, size=n_argo_days, replace=False))
    argo_temp = temp[argo_idx].copy()
    argo_temp += rng.normal(0, 0.25, size=argo_temp.shape).astype(np.float32)  # ARGO obs error

    # ARGO floats don't cover every grid cell every day -- subsample spatially too
    argo_obs_mask = rng.random(argo_temp.shape) < 0.35
    argo_temp[~argo_obs_mask] = np.nan

    argo_ds = xr.Dataset(
        {"temperature": (["time", "depth", "lat", "lon"], argo_temp)},
        coords={"time": dates[argo_idx], "depth": depths, "lat": lat, "lon": lon},
    )
    os.makedirs(C.RAW_DIR, exist_ok=True)
    argo_ds.to_netcdf(C.ARGO_NC_PATH)
    print(f"Wrote synthetic independent ARGO obs -> {C.ARGO_NC_PATH}  "
          f"({n_argo_days} days, ~35% spatial coverage)")

    return surface_path, target_path, C.ARGO_NC_PATH


if __name__ == "__main__":
    generate(n_days=730)
