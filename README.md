# Satellite Embedding-Based Deep Learning Framework for Subsurface Ocean Temperature Reconstruction

Reconstructs 3D subsurface ocean temperature (15 standard depth levels, 0–1000 m) from daily
surface satellite observations, at 0.25° resolution over the **North Indian Ocean**
(5°N–30°N, 45°E–105°E).

## Architecture: Attention Residual U-Net

Surface inputs (SST, SSS, SSH/SLA, surface currents U/V, surface winds U/V, stacked over a
configurable trailing time window) are encoded through a 4-level residual CNN encoder into a
compact bottleneck **"satellite embedding"**, then decoded back to full grid resolution through
an attention-gated decoder that produces the full depth-wise temperature field in one forward
pass.

```
surface fields (T x 7, H, W)
        │
   ResidualBlock (stem)
        │
  ┌── Down1 ── Down2 ── Down3 ── Down4 ──┐
  │                                       │
  skip1    skip2    skip3    skip4    Bottleneck (= satellite embedding, 256-d)
  │                                       │
  └── Up1 ←── Up2 ←── Up3 ←── Up4 ←──────┘
        │      (attention-gated skip connections)
   1x1 conv head → per-depth scale/bias
        │
temperature (15 depths, H, W)
```

Why this design (see `model.py` docstring for full rationale):
- **U-Net encoder-decoder** — required because output is dense per-pixel regression at input
  resolution, not a single label (rules out a plain CNN → global-pool → FC head).
- **Residual blocks** — stable gradients at moderate depth on comparatively small daily-ocean
  datasets (years, not millions of images).
- **Attention gates on skip connections** — down-weight land / noisy coastal retrievals, focus
  decoder capacity on dynamically active regions (fronts, eddies, upwelling).
- **Bottleneck = embedding** — exposed via `model.encode()`, directly satisfies the "generate
  compact satellite embeddings" requirement.
- **Learnable per-depth scale/bias** — initializes output close to a sensible warm-surface /
  cold-deep climatology, which speeds convergence substantially vs. zero-init.

## Project layout

```
config.py                  All paths, domain grid, depth levels, hyperparameters
data_pipeline.py           Regridding/harmonization + PyTorch Dataset
model.py                   AttentionResUNet + masked depth-weighted loss
train.py                   Training loop (early stopping, LR scheduling, checkpointing)
evaluate.py                Skill metrics (RMSE/bias/corr per depth) incl. independent ARGO check
utils.py                   Masked metrics, seeding, EarlyStopper
generate_synthetic_data.py Physically-plausible synthetic data for full pipeline dry-runs
requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Sanity-check the model builds and shapes are correct
python model.py

# 2. (Optional) generate a synthetic dataset to validate the FULL pipeline
#    end-to-end before plugging in real data:
python generate_synthetic_data.py     # writes to data/interim/, data/raw/

# 3. Train
python train.py --surface data/interim/surface_daily.nc \
                 --target  data/interim/glorys_temperature.nc

# 4. Evaluate (test split + independent Gridded ARGO validation)
python evaluate.py --surface data/interim/surface_daily.nc \
                    --target  data/interim/glorys_temperature.nc \
                    --argo    data/raw/gridded_argo_temperature.nc
```

## Using real data

Swap in real files and point `data_pipeline.build_daily_stack()` at your sources:

| Variable | Suggested source |
|---|---|
| SST | OSTIA / GHRSST / MUR |
| SSS | SMAP / SMOS / OISSS |
| SSH / SLA | CMEMS / AVISO altimetry |
| Surface currents | OSCAR, or geostrophic from SSH |
| Surface winds | CCMP, ERA5 |
| Target temperature | **GLORYS reanalysis** — `https://doi.org/10.48670/moi-00021` |
| Independent validation | **Gridded ARGO — INCOIS Live Access Server (LAS)** |

`data_pipeline.harmonize_dataset()` regrids each raw source (native grid/resolution) onto the
common 0.25° daily North Indian Ocean grid — using `xesmf` conservative/bilinear regridding if
installed, otherwise falling back to `xarray` linear interpolation (fine for products already on
a regular lat/lon grid). Missing/land cells are tracked via an explicit mask that is respected
throughout training and evaluation (loss, RMSE, correlation, bias are all mask-aware — land or
missing-observation pixels never contaminate the metrics).

## Notes on the synthetic data generator

`generate_synthetic_data.py` is **only for validating the pipeline mechanics** (shapes, masking,
train/val/test splitting, checkpointing, ARGO comparison logic) before you have real files. It
builds a toy ocean where SSH-driven "eddies" physically displace a warm-surface/cold-deep
vertical temperature profile — the same physical relationship (thermocline tilt ↔ surface
signature) the real model is meant to learn from actual GLORYS/satellite data — plus a
seasonal cycle and a simple monsoon wind proxy. It is not a substitute for real oceanographic
data in a genuine PoC.

## Extending

- Swap `AttentionResUNet` for a ConvLSTM / temporal-attention variant if you want explicit
  sequence modeling instead of simple channel-stacking of `TIME_WINDOW` days (`config.py`).
- `model.encode()` exposes the bottleneck embedding directly for diagnostics — e.g. clustering
  daily "ocean states", or as a feature for other downstream models (mixed-layer depth,
  isotherm depth, heat content).
- `utils.per_depth_metrics()` returns per-depth RMSE/bias/correlation as a DataFrame-ready list;
  hook this up to your own plotting for depth-vs-skill and Hovmöller-style diagnostics.
