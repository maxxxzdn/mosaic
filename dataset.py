"""Metadata dataclasses for weather forecasting inference."""

import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizationStats:
    """Normalization statistics for state variables."""
    state_mean: torch.Tensor
    state_std: torch.Tensor
    residual_mean: torch.Tensor
    residual_std: torch.Tensor

    def to(self, device) -> 'NormalizationStats':
        return NormalizationStats(
            state_mean=self.state_mean.to(device),
            state_std=self.state_std.to(device),
            residual_mean=self.residual_mean.to(device),
            residual_std=self.residual_std.to(device),
        )


@dataclass(frozen=True)
class WeatherMetadata:
    """Metadata for the weather dataset."""
    variables: list[str]
    static_variables: list[str]
    longitude: torch.Tensor
    latitude: torch.Tensor
    static_data: torch.Tensor
    day_year_delta: torch.Tensor
    norm_stats: NormalizationStats
