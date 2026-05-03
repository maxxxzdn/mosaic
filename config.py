# Variable names and pressure levels for the U-NBSA weather forecasting model.

SL_VARS: list[str] = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]

PL_VARS: list[str] = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]

ST_VARS: list[str] = [
    "geopotential_at_surface",
    "land_sea_mask",
    "soil_type",
]

LEVELS: list[int] = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
