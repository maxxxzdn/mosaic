"""
Run autoregressive global weather forecasts with the U-NBSA 1.5° model.

The model predicts 6-hourly atmospheric states autoregressively, supporting
both deterministic (1 member) and probabilistic (N members) forecasts.

Usage:
    python inference.py \\
        --checkpoint 12_best.pt \\
        --zarr /path/to/era5.zarr \\
        --init-time "2020-01-01T00:00" \\
        --steps 40 \\
        --members 1 \\
        --output forecast.npz

    # Using WeatherBench2 GCS data (requires gcsfs):
    python inference.py \\
        --checkpoint 12_best.pt \\
        --zarr gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr \\
        --init-time "2022-01-01T00:00" \\
        --steps 40 \\
        --members 1 \\
        --output forecast.npz

Input zarr format:
    The zarr store must contain the following variables at 1.5° resolution
    (240 lon × 121 lat, 6-hourly timesteps):
    - Surface:  2m_temperature, 10m_u_component_of_wind, 10m_v_component_of_wind,
                mean_sea_level_pressure
    - Pressure-level (at 13 levels 50..1000 hPa): geopotential, specific_humidity,
                temperature, u_component_of_wind, v_component_of_wind, vertical_velocity
    - Coordinates: longitude (240,), latitude (121,), time (hours since 1959-01-01)

Output npz:
    forecasts       float32 (members, steps, 240, 121, 82) – physical units
    variables       list of 82 variable names
    lead_time_hours int32   (steps,)  – [6, 12, ..., steps*6]
    init_time       str     – initialization timestamp

Hardware:
    Requires a CUDA GPU. An A100 (80 GB) is recommended for multi-member ensembles.
    float16 inference is used by default (~20-25 GB for 1 member, 40-step rollout).
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

# ---------------------------------------------------------------------------
# Model imports
# ---------------------------------------------------------------------------
from config import SL_VARS, PL_VARS, ST_VARS, LEVELS
from dataset import NormalizationStats, WeatherMetadata
from mosaic import Transformer, ModelConfig, StageConfig, BottleneckConfig
from base import WeatherModel

# ---------------------------------------------------------------------------
# Hardcoded model config (matches checkpoints/finetune_15deg_norm/12_best.pt)
# ---------------------------------------------------------------------------

STEP_STRIDE = 1          # 6h per step
NUM_HISTORY_STEPS = 4    # 24h of input history
DTYPE = torch.float16

_STAGE_CFGS = [
    StageConfig(
        nside=64, dim=768, num_heads=12,
        block_attn_size=1024, sparse_block_size=128, sparse_block_count=24,
        encoder_depth=4, decoder_depth=2, mlp_ratio=4.0, gqa_ratio=4,
    ),
    StageConfig(
        nside=32, dim=1024, num_heads=16,
        block_attn_size=1024, sparse_block_size=128, sparse_block_count=12,
        encoder_depth=4, decoder_depth=2, mlp_ratio=4.0, gqa_ratio=4,
    ),
]

_BOTTLENECK_CFG = BottleneckConfig(
    nside=16, dim=1280, num_heads=20,
    block_attn_size=1024, sparse_block_size=128, sparse_block_count=4,
    depth=2, mlp_ratio=4.0, gqa_ratio=4,
)


# ---------------------------------------------------------------------------
# Time utilities
# ---------------------------------------------------------------------------

def compute_day_year_progress(timestamp: pd.Timestamp):
    """Return (day_progress, year_progress) fractions for a single timestamp."""
    day_progress = (timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second) / 86400.0
    days_in_year = 366 if timestamp.is_leap_year else 365
    year_progress = (timestamp.day_of_year - 1) / days_in_year
    return float(day_progress), float(year_progress)


# ---------------------------------------------------------------------------
# Zarr loading
# ---------------------------------------------------------------------------

def _load_zarr_times(store) -> pd.DatetimeIndex:
    """Load and decode the time coordinate from the zarr store."""
    time_raw = np.asarray(store['time'])
    # Try integer (hours since 1959-01-01) then string/datetime64
    if np.issubdtype(time_raw.dtype, np.integer):
        return pd.to_datetime(time_raw, unit='h', origin='1959-01-01')
    return pd.to_datetime(time_raw)


def load_initial_state(zarr_path: str, init_time: str, num_history_steps: int = 4):
    """
    Load `num_history_steps` timesteps ending at `init_time` from a zarr store.

    Returns:
        state: np.ndarray of shape (num_history_steps, 240, 121, 82) in physical units
        day_year_time: tuple (day_progress, year_progress) for init_time
        longitude: np.ndarray (240,)
        latitude: np.ndarray (121,) South→North
    """
    # Open zarr (supports local paths, gs://, s3://, etc.)
    if zarr_path.startswith('gs://'):
        import gcsfs
        fs = gcsfs.GCSFileSystem(token='anon')
        store_obj = zarr.open(fs.get_mapper(zarr_path), mode='r')
    else:
        store_obj = zarr.open(zarr_path, mode='r')

    times = _load_zarr_times(store_obj)
    init_ts = pd.Timestamp(init_time)

    # Find the index of init_time
    idx = times.searchsorted(init_ts)
    if idx >= len(times) or times[idx] != init_ts:
        raise ValueError(
            f"init_time '{init_time}' not found in zarr store. "
            f"Available range: {times[0]} to {times[-1]}"
        )

    start_idx = idx - num_history_steps + 1
    if start_idx < 0:
        raise ValueError(
            f"Not enough history: need {num_history_steps} steps before {init_time}, "
            f"but data starts at {times[0]}"
        )

    # Load longitude/latitude
    longitude = np.asarray(store_obj['longitude'])           # (240,) 0..358.5
    latitude_raw = np.asarray(store_obj['latitude'])         # (121,) North→South in ERA5/HRES
    latitude = latitude_raw[::-1].copy()                     # flip to South→North

    n_lon, n_lat = len(longitude), len(latitude)
    n_vars = len(SL_VARS) + len(PL_VARS) * len(LEVELS)
    state = np.empty((num_history_steps, n_lon, n_lat, n_vars), dtype=np.float32)

    for step_i, t_idx in enumerate(range(start_idx, start_idx + num_history_steps)):
        ch = 0
        # Surface variables: (lat, lon) → flip lat → transpose to (lon, lat)
        for var in SL_VARS:
            arr = np.asarray(store_obj[var][t_idx])          # (lat, lon)
            arr = arr[::-1, :]                               # flip lat to S→N
            state[step_i, :, :, ch] = arr.T                 # transpose to (lon, lat)
            ch += 1

        # Pressure-level variables: (level, lat, lon) → select 13 levels → flip lat → transpose
        all_levels_in_store = list(np.asarray(store_obj['level'])) if 'level' in store_obj else None
        for var in PL_VARS:
            arr_full = np.asarray(store_obj[var][t_idx])     # (all_levels, lat, lon)
            for level in LEVELS:
                if all_levels_in_store is not None:
                    lev_idx = all_levels_in_store.index(level)
                    arr = arr_full[lev_idx]                  # (lat, lon)
                else:
                    arr = arr_full[LEVELS.index(level)]
                arr = arr[::-1, :]                           # flip lat to S→N
                state[step_i, :, :, ch] = arr.T             # (lon, lat)
                ch += 1

    day_progress, year_progress = compute_day_year_progress(init_ts)
    return state, (day_progress, year_progress), longitude, latitude


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(
    checkpoint_path: str,
    variables: list,
    longitude: np.ndarray,
    latitude: np.ndarray,
    norm_stats_path: str = "norm_stats.npz",
    static_vars_path: str = "static_vars.npz",
    device: str = "cuda",
):
    """Build and return the WeatherModel with loaded checkpoint and metadata."""

    # Load normalization statistics
    _ns = np.load(norm_stats_path)
    norm_stats = NormalizationStats(
        state_mean=torch.from_numpy(_ns['state_mean'].astype(np.float32)),
        state_std=torch.from_numpy(_ns['state_std'].astype(np.float32)),
        residual_mean=torch.from_numpy(_ns['residual_mean'].astype(np.float32)) if 'residual_mean' in _ns else torch.zeros(len(variables)),
        residual_std=torch.from_numpy(_ns['residual_std'].astype(np.float32)) if 'residual_std' in _ns else torch.ones(len(variables)),
    )

    # Load static variables
    _sv = np.load(static_vars_path)
    static_data = torch.from_numpy(_sv['data'].astype(np.float32))    # (lon, lat, 3)
    lon_tensor = torch.from_numpy(longitude.astype(np.float32))
    lat_tensor = torch.from_numpy(latitude.astype(np.float32))

    day_year_delta = torch.tensor(
        [STEP_STRIDE / 4.0, STEP_STRIDE / 365.25], dtype=torch.float32
    )

    metadata = WeatherMetadata(
        variables=variables,
        static_variables=list(ST_VARS),
        longitude=lon_tensor,
        latitude=lat_tensor,
        static_data=static_data,
        day_year_delta=day_year_delta,
        norm_stats=norm_stats,
    )

    # Build model
    model_config = ModelConfig(
        dim=_STAGE_CFGS[0].dim,
        num_heads=_STAGE_CFGS[0].num_heads,
        variables=variables,
        static_variables=list(ST_VARS),
        k_neighbors=20,
        qk_norm=False,
        rope=True,
        rope_theta=10000,
        nsa_every=1,
        qkv_compress_ratio=1,
        num_history_steps=NUM_HISTORY_STEPS,
        noise_dim=32,
        rmsnorm_elementwise_affine=False,
        cg_stage_cfgs=_STAGE_CFGS,
        bottleneck_cfg=_BOTTLENECK_CFG,
    )

    backbone = Transformer(model_config)
    model = WeatherModel(backbone, metadata).to(device).eval()

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    return model, metadata


# ---------------------------------------------------------------------------
# Autoregressive rollout (direct state prediction)
# ---------------------------------------------------------------------------

@torch.no_grad()
def unroll_direct(
    model: WeatherModel,
    initial_unnorm_state: torch.Tensor,
    day_year_time: torch.Tensor,
    day_year_delta: torch.Tensor,
    norm_stats: NormalizationStats,
    num_unroll_steps: int,
    num_ensemble_members: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Autoregressively forecast using direct state prediction (learn_direct=True).

    Args:
        model: WeatherModel
        initial_unnorm_state: (B, num_history_steps, lon, lat, channels) in physical units
        day_year_time: (B, 2) day/year progress fractions at init_time
        day_year_delta: (2,) increment per step
        norm_stats: NormalizationStats on the target device
        num_unroll_steps: number of 6-hourly steps to forecast
        num_ensemble_members: number of ensemble members (noise samples on step 0)
        dtype: computation dtype (float16 recommended)

    Returns:
        trajectory: (B, members, num_history_steps + num_unroll_steps, lon, lat, channels)
    """
    batch_size = initial_unnorm_state.shape[0]
    num_history_steps = initial_unnorm_state.shape[1]
    device = initial_unnorm_state.device

    trajectory = torch.empty(
        (batch_size, num_ensemble_members, num_unroll_steps + num_history_steps)
        + initial_unnorm_state.shape[2:],
        dtype=initial_unnorm_state.dtype,
        device=device,
    )

    # Expand initial state to ensemble dimension
    current_unnorm_state = initial_unnorm_state.unsqueeze(1)   # (B, 1, H, lon, lat, C)
    current_day_year_time = day_year_time.unsqueeze(1)          # (B, 1, 2)

    trajectory[:, :, :num_history_steps] = current_unnorm_state

    for t in range(num_unroll_steps):
        # Expand ensemble only on the first step
        num_ens_step = num_ensemble_members if t == 0 else 1

        current_norm_state = (current_unnorm_state - norm_stats.state_mean) / norm_stats.state_std
        with torch.amp.autocast('cuda', dtype=dtype):
            norm_next_state = model(current_norm_state, current_day_year_time, num_ens_step)

        next_unnorm_state = norm_next_state * norm_stats.state_std + norm_stats.state_mean
        current_day_year_time = current_day_year_time + day_year_delta.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_ens_step, -1
        )

        trajectory[:, :, t + num_history_steps] = next_unnorm_state
        current_unnorm_state = trajectory[:, :, t + 1 : t + 1 + num_history_steps]

    return trajectory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="U-NBSA 1.5° Weather Forecast Inference")
    parser.add_argument("--checkpoint", type=str, default="12_best.pt",
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--zarr", type=str, required=True,
                        help="Path or GCS URI to zarr store with ERA5/HRES data at 1.5°")
    parser.add_argument("--init-time", type=str, required=True,
                        help="Initialization time (ISO 8601), e.g. '2020-01-01T00:00'")
    parser.add_argument("--steps", type=int, default=40,
                        help="Number of 6-hourly forecast steps (default: 40 = 10 days)")
    parser.add_argument("--members", type=int, default=1,
                        help="Number of ensemble members (default: 1)")
    parser.add_argument("--output", type=str, default="forecast.npz",
                        help="Output file path (default: forecast.npz)")
    parser.add_argument("--norm-stats", type=str, default="norm_stats.npz",
                        help="Path to norm_stats.npz (default: norm_stats.npz in current dir)")
    parser.add_argument("--static-vars", type=str, default="static_vars.npz",
                        help="Path to static_vars.npz (default: static_vars.npz in current dir)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (slower but easier to debug)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    args = parser.parse_args()

    device = args.device
    torch.set_float32_matmul_precision('high')

    # Build variable list: 4 surface + 6*13 pressure-level = 82 channels
    variables = list(SL_VARS)
    for var in PL_VARS:
        for level in LEVELS:
            variables.append(f"{var}_{level}")
    print(f"Variables: {len(variables)} channels")

    # Load initial state from zarr
    print(f"Loading initial state from zarr: {args.zarr}")
    print(f"  Init time: {args.init_time}  (history: {NUM_HISTORY_STEPS} x 6h steps)")
    initial_state_np, (day_prog, year_prog), longitude, latitude = load_initial_state(
        args.zarr, args.init_time, NUM_HISTORY_STEPS
    )
    print(f"  State shape: {initial_state_np.shape}  (steps, lon, lat, channels)")

    # Build model and load checkpoint
    print(f"\nBuilding model and loading checkpoint: {args.checkpoint}")
    model, metadata = build_model(
        checkpoint_path=args.checkpoint,
        variables=variables,
        longitude=longitude,
        latitude=latitude,
        norm_stats_path=args.norm_stats,
        static_vars_path=args.static_vars,
        device=device,
    )
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {num_params:.1f}M")

    # Optionally compile
    if not args.no_compile:
        print("Compiling model with torch.compile (reduce-overhead)...")
        unroll_fn = torch.compile(unroll_direct, mode='reduce-overhead')
    else:
        unroll_fn = unroll_direct

    # Prepare tensors
    initial_state = torch.from_numpy(initial_state_np).unsqueeze(0).to(device)  # (1, H, lon, lat, C)
    day_year_time = torch.tensor([[day_prog, year_prog]], dtype=torch.float32, device=device)  # (1, 2)
    norm_stats_d = metadata.norm_stats.to(device)
    day_year_delta_d = metadata.day_year_delta.to(device)

    # Run forecast
    print(f"\nRunning {args.steps}-step forecast ({args.steps * 6}h) with {args.members} member(s)...")
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        trajectory = unroll_fn(
            model=model,
            initial_unnorm_state=initial_state,
            day_year_time=day_year_time,
            day_year_delta=day_year_delta_d,
            norm_stats=norm_stats_d,
            num_unroll_steps=args.steps,
            num_ensemble_members=args.members,
            dtype=DTYPE,
        )

    if device == 'cuda':
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory: {peak_gb:.1f} GB")

    # Extract forecast steps (skip history)
    forecasts = trajectory[0, :, NUM_HISTORY_STEPS:].cpu().numpy()  # (members, steps, lon, lat, C)
    print(f"  Forecast shape: {forecasts.shape}")

    # Save output
    lead_time_hours = np.arange(1, args.steps + 1) * 6
    np.savez(
        args.output,
        forecasts=forecasts,
        variables=np.array(variables),
        lead_time_hours=lead_time_hours,
        init_time=np.str_(args.init_time),
        longitude=longitude,
        latitude=latitude,
    )
    print(f"\nSaved forecast to: {args.output}")
    print(f"  Shape: forecasts {forecasts.shape}  (members, steps, lon=240, lat=121, channels=82)")
    print(f"  Lead times: {lead_time_hours[0]}h to {lead_time_hours[-1]}h")


if __name__ == "__main__":
    main()
