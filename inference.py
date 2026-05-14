"""
Run autoregressive global weather forecasts with the Mosaic 1.5° model.

The model predicts 6-hourly atmospheric states autoregressively, supporting
both deterministic (1 member) and probabilistic (N members) forecasts.

Usage:
    # ERA5 variant (24h steps), default checkpoint and norm stats inferred from --variant
    python inference.py --variant era5 \\
        --zarr gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr \\
        --init-time "2020-01-01T00:00" --steps 10 --output forecast_era5.npz

    # HRES variant (6h steps)
    python inference.py --variant hres \\
        --zarr gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr \\
        --init-time "2022-01-01T00:00" --steps 40 --output forecast_hres.npz

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
    lead_time_hours int32   (steps,)  – multiples of step_stride*6h
                                       (era5: 24, 48, ...; hres: 6, 12, ...)
    init_time       str     – initialization timestamp

Hardware:
    Requires a CUDA GPU. A 16 GB GPU is enough for 1 member; A100 80 GB recommended
    for multi-member ensembles. float16 inference (~9 GB for 1 member, 40-step rollout).
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
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

DTYPE = torch.float16

# ---------------------------------------------------------------------------
# Model variant presets
# ---------------------------------------------------------------------------
# The two published variants share the same Mosaic architecture (stage / bottleneck
# sizes) but differ in training data, time cadence, history length, neighbour
# count, and normalisation statistics:
#   - `era5`: ERA5-only training, 24h steps (4 x 6h), 2 input states, k=24 neighbours
#   - `hres`: ERA5 pretrain + HRES finetune, 6h steps, 4 input states, k=20 neighbours
# ---------------------------------------------------------------------------

_STAGE_CFGS_COMMON = [
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

_BOTTLENECK_CFG_COMMON = BottleneckConfig(
    nside=16, dim=1280, num_heads=20,
    block_attn_size=1024, sparse_block_size=128, sparse_block_count=4,
    depth=2, mlp_ratio=4.0, gqa_ratio=4,
)


@dataclass
class Preset:
    step_stride: int            # number of native 6h timesteps per model step
    num_history_steps: int      # number of input states fed to the model
    k_neighbors: int            # neighbours used in cross-attention interpolation
    default_checkpoint: str
    default_norm_stats: str
    stage_cfgs: list
    bottleneck_cfg: BottleneckConfig


PRESETS = {
    "era5": Preset(
        step_stride=4, num_history_steps=2, k_neighbors=24,
        default_checkpoint="checkpoints/era5_best.pt",
        default_norm_stats="data/norm_stats_era5.npz",
        stage_cfgs=_STAGE_CFGS_COMMON,
        bottleneck_cfg=_BOTTLENECK_CFG_COMMON,
    ),
    "hres": Preset(
        step_stride=1, num_history_steps=4, k_neighbors=20,
        default_checkpoint="checkpoints/hres_best.pt",
        default_norm_stats="data/norm_stats_hres.npz",
        stage_cfgs=_STAGE_CFGS_COMMON,
        bottleneck_cfg=_BOTTLENECK_CFG_COMMON,
    ),
}


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
    """Load and decode the time coordinate from the zarr store, honouring its units attr."""
    time_raw = np.asarray(store['time'])
    if not np.issubdtype(time_raw.dtype, np.integer):
        return pd.to_datetime(time_raw)
    # Integer encoding: parse 'units' attr e.g. "hours since 1959-01-01"
    units = store['time'].attrs.get('units', 'hours since 1959-01-01')
    try:
        unit_word, _, origin = units.partition(' since ')
    except Exception:
        unit_word, origin = 'hours', '1959-01-01'
    unit_map = {'hours': 'h', 'hour': 'h', 'days': 'D', 'day': 'D',
                'minutes': 'm', 'minute': 'm', 'seconds': 's', 'second': 's'}
    unit = unit_map.get(unit_word.strip().lower(), 'h')
    return pd.to_datetime(time_raw, unit=unit, origin=origin.strip() or '1959-01-01')


def load_initial_state(zarr_path: str, init_time: str, num_history_steps: int = 4, step_stride: int = 1):
    """
    Load `num_history_steps` timesteps ending at `init_time` from a zarr store,
    spaced `step_stride * 6h` apart (so step_stride=4 -> 24h spacing).

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

    # history indices: [idx - (H-1)*S, idx - (H-2)*S, ..., idx]
    history_indices = [idx - (num_history_steps - 1 - i) * step_stride for i in range(num_history_steps)]
    if history_indices[0] < 0:
        raise ValueError(
            f"Not enough history: need {num_history_steps} steps spaced {step_stride*6}h apart "
            f"before {init_time}, but data starts at {times[0]}"
        )

    # Load longitude/latitude in the canonical (lon, lat S→N) order the model expects.
    longitude = np.asarray(store_obj['longitude'])           # (240,) 0..358.5
    latitude_raw = np.asarray(store_obj['latitude'])         # (121,)
    if latitude_raw[0] > latitude_raw[-1]:                   # N→S in store → flip
        latitude = latitude_raw[::-1].copy()
        flip_lat = True
    else:
        latitude = latitude_raw.copy()
        flip_lat = False

    n_lon, n_lat = len(longitude), len(latitude)
    n_vars = len(SL_VARS) + len(PL_VARS) * len(LEVELS)
    state = np.empty((num_history_steps, n_lon, n_lat, n_vars), dtype=np.float32)

    def _to_lon_lat(arr: np.ndarray, dims: list) -> np.ndarray:
        """Normalise a (lat,lon) or (lon,lat) slice to (lon, lat S→N)."""
        if dims[-2:] == ['latitude', 'longitude']:
            arr = arr.T                                       # (lat,lon) -> (lon,lat)
        elif dims[-2:] != ['longitude', 'latitude']:
            raise ValueError(f"unexpected dim order: {dims}")
        if flip_lat:
            arr = arr[:, ::-1]
        return np.ascontiguousarray(arr)

    all_levels_in_store = list(np.asarray(store_obj['level'])) if 'level' in store_obj else None

    for step_i, t_idx in enumerate(history_indices):
        ch = 0
        for var in SL_VARS:
            dims = list(store_obj[var].attrs.get('_ARRAY_DIMENSIONS', ['time', 'latitude', 'longitude']))
            arr = np.asarray(store_obj[var][t_idx])          # 2D
            state[step_i, :, :, ch] = _to_lon_lat(arr, dims)
            ch += 1

        for var in PL_VARS:
            dims = list(store_obj[var].attrs.get('_ARRAY_DIMENSIONS', ['time', 'level', 'latitude', 'longitude']))
            arr_full = np.asarray(store_obj[var][t_idx])     # 3D (level, ...)
            spatial_dims = [d for d in dims if d != 'time']  # drop time (already indexed)
            for level in LEVELS:
                lev_idx = all_levels_in_store.index(level) if all_levels_in_store is not None else LEVELS.index(level)
                arr = arr_full[lev_idx]                       # 2D
                # spatial_dims still includes 'level' at the front; pass just the 2D part
                state[step_i, :, :, ch] = _to_lon_lat(arr, spatial_dims[1:] if spatial_dims[0] == 'level' else spatial_dims)
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
    preset: Preset,
    norm_stats_path: str = "data/norm_stats_era5.npz",
    static_vars_path: str = "data/static_vars.npz",
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
        [preset.step_stride / 4.0, preset.step_stride / 365.25], dtype=torch.float32
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
        dim=preset.stage_cfgs[0].dim,
        num_heads=preset.stage_cfgs[0].num_heads,
        variables=variables,
        static_variables=list(ST_VARS),
        k_neighbors=preset.k_neighbors,
        qk_norm=False,
        rope=True,
        rope_theta=10000,
        sparse_every=1,
        qkv_compress_ratio=1,
        num_history_steps=preset.num_history_steps,
        noise_dim=32,
        rmsnorm_elementwise_affine=False,
        cg_stage_cfgs=preset.stage_cfgs,
        bottleneck_cfg=preset.bottleneck_cfg,
    )

    backbone = Transformer(model_config)
    model = WeatherModel(backbone, metadata).to(device).eval()

    # Load checkpoint. The model registers several deterministic buffers (RoPE
    # tables, HEALPix neighbour indices, static_vars) that are recomputed at
    # __init__ from the metadata/config and therefore aren't expected in the
    # saved checkpoint — so we load non-strictly and only warn on *unexpected*
    # keys, which would indicate a real architecture mismatch.
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys in checkpoint (architecture mismatch): {result.unexpected_keys[:5]}"
            + (f" ... and {len(result.unexpected_keys)-5} more" if len(result.unexpected_keys) > 5 else "")
        )
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    print(f"  {len(state_dict)} keys loaded, {len(result.missing_keys)} buffer keys re-computed from config")

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
    parser = argparse.ArgumentParser(description="Mosaic 1.5° Weather Forecast Inference")
    parser.add_argument("--variant", type=str, required=True, choices=sorted(PRESETS.keys()),
                        help="Model variant: 'era5' (ERA5-only, 24h steps) or 'hres' (ERA5+HRES finetune, 6h steps)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt). Default: preset's default_checkpoint")
    parser.add_argument("--zarr", type=str, required=True,
                        help="Path or GCS URI to zarr store with ERA5/HRES data at 1.5°")
    parser.add_argument("--init-time", type=str, required=True,
                        help="Initialization time (ISO 8601), e.g. '2020-01-01T00:00'")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of forecast steps (each step = step_stride*6h; e.g. era5 step=24h, hres step=6h)")
    parser.add_argument("--members", type=int, default=1,
                        help="Number of ensemble members (default: 1)")
    parser.add_argument("--output", type=str, default="forecast.npz",
                        help="Output file path (default: forecast.npz)")
    parser.add_argument("--norm-stats", type=str, default=None,
                        help="Path to norm_stats .npz. Default: preset's default_norm_stats")
    parser.add_argument("--static-vars", type=str, default="data/static_vars.npz",
                        help="Path to static_vars.npz (default: data/static_vars.npz)")
    parser.add_argument("--k-neighbors", type=int, default=None,
                        help="Override preset's k_neighbors (advanced — for ablation only)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (slower but easier to debug)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    args = parser.parse_args()

    preset = PRESETS[args.variant]
    if args.k_neighbors is not None and args.k_neighbors != preset.k_neighbors:
        from dataclasses import replace
        preset = replace(preset, k_neighbors=args.k_neighbors)
    checkpoint_path = args.checkpoint or preset.default_checkpoint
    norm_stats_path = args.norm_stats or preset.default_norm_stats
    print(f"Variant: {args.variant}  "
          f"(step_stride={preset.step_stride}, num_history_steps={preset.num_history_steps}, "
          f"k_neighbors={preset.k_neighbors})")

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
    print(f"  Init time: {args.init_time}  (history: {preset.num_history_steps} x {preset.step_stride*6}h steps)")
    initial_state_np, (day_prog, year_prog), longitude, latitude = load_initial_state(
        args.zarr, args.init_time,
        num_history_steps=preset.num_history_steps,
        step_stride=preset.step_stride,
    )
    print(f"  State shape: {initial_state_np.shape}  (steps, lon, lat, channels)")

    # Build model and load checkpoint
    print(f"\nBuilding model and loading checkpoint: {checkpoint_path}")
    model, metadata = build_model(
        checkpoint_path=checkpoint_path,
        variables=variables,
        longitude=longitude,
        latitude=latitude,
        preset=preset,
        norm_stats_path=norm_stats_path,
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
    total_hours = args.steps * preset.step_stride * 6
    print(f"\nRunning {args.steps}-step forecast ({total_hours}h) with {args.members} member(s)...")
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
    forecasts = trajectory[0, :, preset.num_history_steps:].cpu().numpy()  # (members, steps, lon, lat, C)
    print(f"  Forecast shape: {forecasts.shape}")

    # Save output
    lead_time_hours = np.arange(1, args.steps + 1) * 6 * preset.step_stride
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
