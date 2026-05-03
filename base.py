import torch
from torch import nn
from dataset import WeatherMetadata


class WeatherModel(nn.Module):
    """Weather forecasting model wrapper."""

    def __init__(self, model: nn.Module, weather_metadata: WeatherMetadata):
        super().__init__()
        self.model = model
        self.model.initialize_static_vars(weather_metadata.static_data, weather_metadata.longitude, weather_metadata.latitude)
        self.model.initialize_interpolation(weather_metadata.longitude, weather_metadata.latitude)
        self.weather_metadata = weather_metadata

    def forward(self, norm_state: torch.Tensor, day_year_time: torch.Tensor, num_noise_samples: int):
        return self.model(norm_state, day_year_time, num_noise_samples)
