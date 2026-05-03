# U-NBSA: Sparse Attention Weather Forecasting Model (1.5°)

Global weather forecasting model at 1.5° resolution (240×121 grid), producing 6-hourly forecasts up to 10 days (40 steps).

## Model Description

**U-NBSA** (U-Net Native Block Sparse Attention) is a transformer-based weather forecasting model that operates on a spherical grid. Key architectural features:

- **Grid**: 1.5° equiangular global grid (240 lon × 121 lat = 29,040 points)
- **Input**: 4 consecutive 6-hourly states (24h history), 82 atmospheric channels
- **Output**: Autoregressive 6-hourly predictions via direct state prediction
- **Attention**: Native Block Sparse Attention (NBSA) combining local block attention, mean-pooled compression attention, and top-k sparse selection, computed on a HEALPix grid for better spatial locality
- **Architecture**: U-Net encoder–bottleneck–decoder with HEALPix up/downsampling and skip connections
- **Ensemble**: Probabilistic forecasts via learned noise injection (generates N members from a single initial state)

### Variables (82 channels)

**Surface (4):** `2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, `mean_sea_level_pressure`

**Pressure-level (6 × 13 = 78):** `geopotential`, `specific_humidity`, `temperature`, `u_component_of_wind`, `v_component_of_wind`, `vertical_velocity` at levels [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] hPa

**Static (3, not in output):** `geopotential_at_surface`, `land_sea_mask`, `soil_type`

### Architecture Details

| Component | Config |
|-----------|--------|
| Stage 1 | nside=64, dim=768, 12 heads, 4+2 enc/dec layers |
| Stage 2 | nside=32, dim=1024, 16 heads, 4+2 enc/dec layers |
| Bottleneck | nside=16, dim=1280, 20 heads, 2 layers |
| GQA ratio | 4 (12 Q heads → 3 KV heads per stage) |
| Noise dim | 32 (probabilistic ensemble) |
| Parameters | ~800M |

## Hardware Requirements

- **GPU**: CUDA GPU required (A100 80 GB recommended)
- **Memory**: ~20–25 GB GPU RAM for 1-member, 40-step rollout (float16)
- **CUDA**: 11.8+ with matching `triton` and `flash-attn` versions

## Installation

```bash
pip install torch einops healpy scikit-learn numpy zarr pandas triton
pip install flash-attn --no-build-isolation
```

For reading data from Google Cloud Storage:
```bash
pip install gcsfs
```

## Quick Start

```bash
# Deterministic 10-day forecast from HRES 2022 data
python inference.py \
    --checkpoint 12_best.pt \
    --zarr gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
    --init-time "2022-01-01T00:00" \
    --steps 40 \
    --members 1 \
    --output forecast_2022-01-01.npz

# Ensemble forecast (4 members)
python inference.py \
    --checkpoint 12_best.pt \
    --zarr /path/to/local/era5.zarr \
    --init-time "2020-06-15T12:00" \
    --steps 40 \
    --members 4 \
    --output ensemble_forecast.npz
```

## Output Format

The output `.npz` file contains:

| Array | Shape | Description |
|-------|-------|-------------|
| `forecasts` | `(members, steps, 240, 121, 82)` | Predicted states in physical units |
| `variables` | `(82,)` | Variable names (strings) |
| `lead_time_hours` | `(steps,)` | Lead times [6, 12, …, 240] |
| `init_time` | scalar | Initialization timestamp |
| `longitude` | `(240,)` | Longitude values (0 to 358.5°) |
| `latitude` | `(121,)` | Latitude values (South→North, -90 to 90°) |

### Reading the output

```python
import numpy as np

data = np.load("forecast_2022-01-01.npz", allow_pickle=True)
forecasts = data['forecasts']           # (1, 40, 240, 121, 82)
variables = list(data['variables'])     # ['2m_temperature', ...]
lead_hours = data['lead_time_hours']    # [6, 12, ..., 240]

# Extract 500 hPa geopotential at 24h lead time (step index 3 = 24h)
z500_idx = variables.index('geopotential_500')
z500_24h = forecasts[0, 3, :, :, z500_idx]  # (240, 121) lon × lat
```

## Input Data Format

The model accepts ERA5 or HRES data in zarr format at 1.5° resolution with:
- **Grid**: 240 lon × 121 lat equiangular with poles
- **Time**: 6-hourly timesteps encoded as hours since 1959-01-01 (integer) or ISO datetime strings
- **Variables**: All 10 atmospheric variables listed above

Compatible zarr stores from [WeatherBench2](https://weatherbench2.readthedocs.io/):
```
gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr
gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr
```

## Repository Contents

| File | Description |
|------|-------------|
| `inference.py` | Main inference script |
| `unbsa.py` | U-Net NBSA model |
| `nbsa.py` | NBSA attention blocks |
| `ops.py` | Triton sparse attention kernels |
| `utils.py` | HEALPix grid utilities |
| `base.py` | WeatherModel wrapper |
| `config.py` | Variable/level definitions |
| `dataset.py` | Metadata dataclasses |
| `norm_stats.npz` | Normalization statistics (mean/std per variable) |
| `static_vars.npz` | Static variables (orography, land-sea mask, soil type) |
| `12_best.pt` | Trained model checkpoint |

## Citation

If you use this model, please cite:

```bibtex
@inproceedings{zhdanov2025sparse,
  title={(Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models},
  author={Zhdanov, Maksim and Lucic, Ana and Welling, Max and van de Meent, Jan-Willem},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=u0KcfOaRc7}
}
```
