"""
config.py
=========
Central configuration for the Satellite Embedding-Based Deep Learning
Framework for subsurface ocean temperature reconstruction
(North Indian Ocean, 0.25 deg, daily).

Edit paths under DATA PATHS to point at your local / downloaded
NetCDF files (GLORYS, OSTIA-SST, SSS, SSH/AVISO, OSCAR currents,
CCMP/ERA5 winds, Gridded ARGO). Everything else can be left as-is
for a first run.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# DOMAIN (North Indian Ocean: 5N-30N, 45E-105E), 0.25 deg, daily
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
GRID_RES = 0.25

LAT_GRID = np.arange(LAT_MIN, LAT_MAX + GRID_RES, GRID_RES)   # 101 points
LON_GRID = np.arange(LON_MIN, LON_MAX + GRID_RES, GRID_RES)   # 241 points

H = len(LAT_GRID)   # grid height (latitude)
W = len(LON_GRID)   # grid width  (longitude)

# Standard output depth levels (meters) -- 15 levels as specified
DEPTH_LEVELS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
N_DEPTHS = len(DEPTH_LEVELS)

# ---------------------------------------------------------------------------
# INPUT SURFACE VARIABLES (satellite / reanalysis-derived)
# ---------------------------------------------------------------------------
# Each entry: variable_name -> (source description, netcdf variable name)
INPUT_VARIABLES = {
    "sst":        "Sea Surface Temperature (e.g. OSTIA / MUR / GHRSST)",
    "sss":        "Sea Surface Salinity (e.g. SMAP / SMOS / OISSS)",
    "ssh":        "Sea Surface Height / Sea Level Anomaly (e.g. AVISO/CMEMS)",
    "u_curr":     "Surface current, zonal component (e.g. OSCAR / GLORYS)",
    "v_curr":     "Surface current, meridional component (e.g. OSCAR / GLORYS)",
    "u_wind":     "Surface wind, zonal component (e.g. CCMP / ERA5)",
    "v_wind":     "Surface wind, meridional component (e.g. CCMP / ERA5)",
}
N_INPUT_CHANNELS = len(INPUT_VARIABLES)   # 7

# ---------------------------------------------------------------------------
# DATA PATHS  (edit these to your local files)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")            # raw, native-resolution files
INTERIM_DIR = os.path.join(DATA_DIR, "interim")     # regridded/harmonized daily files
PROCESSED_DIR = os.path.join(DATA_DIR, "processed") # final tensors (.npz / .zarr)

TARGET_NC_PATH = os.path.join(RAW_DIR, "glorys_temperature.nc")   # GLORYS reanalysis
ARGO_NC_PATH = os.path.join(RAW_DIR, "gridded_argo_temperature.nc")  # INCOIS LAS gridded ARGO

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

for d in [DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
          CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
# ---------------------------------------------------------------------------
SEED = 42

BATCH_SIZE = 8
NUM_WORKERS = 4
EPOCHS = 150
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 20
LR_SCHEDULER_PATIENCE = 8
LR_SCHEDULER_FACTOR = 0.5
GRAD_CLIP_NORM = 5.0

# Train / val / test split by date (chronological, no shuffling across splits
# to avoid leakage from autocorrelated ocean state)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Model architecture
BASE_CHANNELS = 32          # first-level feature width of the U-Net encoder
EMBEDDING_DIM = 256         # channel depth of the bottleneck "satellite embedding"
USE_ATTENTION_GATES = True
DROPOUT = 0.1

# Sequence context: how many past days of surface fields to feed the model
# (captures mesoscale propagation / mixed-layer memory). 1 = single-day input.
TIME_WINDOW = 3

DEVICE = "cuda"  # falls back to "cpu" automatically at runtime if unavailable
