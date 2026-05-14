"""
Mosaic: U-Net transformer with block-sparse attention for weather forecasting.

Architecture:
- Cross-attention interpolation between lon/lat and HEALPix grids
- Block-sparse attention (local block + compressed + top-k selection branches)
  arranged in a U-Net encoder–bottleneck–decoder
- Probabilistic training with noise injection
"""

import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
from dataclasses import dataclass
from torch.nn import RMSNorm

from utils import get_healpix_grid, rad_to_xyz
from primitives import (
    MosaicBlock as _MosaicBlock,
    CrossAttentionInterpolate,
    NoiseGenerator,
    HEALPixDownsample,
    HEALPixUpsample,
)


@dataclass
class StageConfig:
    """Configuration for a U-Net encoder/decoder stage."""
    nside: int
    dim: int
    num_heads: int
    block_attn_size: int
    sparse_block_size: int
    sparse_block_count: int
    encoder_depth: int
    decoder_depth: int
    mlp_ratio: float
    gqa_ratio: int


@dataclass
class BottleneckConfig:
    """Configuration for the U-Net bottleneck stage."""
    nside: int
    dim: int
    num_heads: int
    block_attn_size: int
    sparse_block_size: int
    sparse_block_count: int
    depth: int
    mlp_ratio: float
    gqa_ratio: int


@dataclass
class ModelConfig:
    """Configuration for the Mosaic model."""
    dim: int
    num_heads: int
    k_neighbors: int
    qk_norm: bool
    rope: bool
    rope_theta: int
    sparse_every: int
    variables: list[str]
    static_variables: list[str]
    qkv_compress_ratio: int
    cg_stage_cfgs: list[StageConfig]
    bottleneck_cfg: BottleneckConfig
    num_history_steps: int = 1
    noise_dim: int = 32
    ortho_init: bool = False
    rmsnorm_elementwise_affine: bool = True
    no_compression: bool = False


@dataclass
class _MergedStageConfig:
    """Merges ModelConfig and StageConfig for compatibility with MosaicBlock."""
    dim: int
    num_heads: int
    block_attn_size: int
    sparse_block_size: int
    sparse_block_count: int
    gqa_ratio: int
    qkv_compress_ratio: int
    rope: bool
    rope_theta: int
    mlp_ratio: float
    noise_dim: int
    rmsnorm_elementwise_affine: bool


def _merge_configs(config: ModelConfig, stage_cfg) -> _MergedStageConfig:
    return _MergedStageConfig(
        dim=stage_cfg.dim,
        num_heads=stage_cfg.num_heads,
        block_attn_size=stage_cfg.block_attn_size,
        sparse_block_size=stage_cfg.sparse_block_size,
        sparse_block_count=stage_cfg.sparse_block_count,
        gqa_ratio=stage_cfg.gqa_ratio,
        qkv_compress_ratio=config.qkv_compress_ratio,
        rope=config.rope,
        rope_theta=config.rope_theta,
        mlp_ratio=stage_cfg.mlp_ratio,
        noise_dim=config.noise_dim,
        rmsnorm_elementwise_affine=config.rmsnorm_elementwise_affine,
    )


def _make_mosaic_block(config: ModelConfig, stage_cfg, block_attn_only: bool) -> _MosaicBlock:
    return _MosaicBlock(_merge_configs(config, stage_cfg), block_attn_only, no_compression=config.no_compression)


class UNetStage(nn.Module):
    def __init__(self, config, stage_cfg, depth):
        super().__init__()
        self.nside = stage_cfg.nside
        self.blocks = nn.ModuleList([
            _make_mosaic_block(
                config=config,
                stage_cfg=stage_cfg,
                block_attn_only=(config.sparse_every <= 0) or not (i % config.sparse_every == 0),
            )
            for i in range(depth)
        ])

    def forward(self, x, z=None):
        for block in self.blocks:
            x = block(x, z)
        return x


class Transformer(nn.Module):
    """U-Net style Transformer for weather forecasting on HEALPix grids."""

    space_dim = 3
    time_dim = 4

    def __init__(self, config: ModelConfig, seed: int = 42):
        super().__init__()

        self.config = config
        self.nside = config.cg_stage_cfgs[0].nside
        self.noise_dim = config.noise_dim

        initial_dim = config.dim
        feature_dim = (len(config.variables) * config.num_history_steps
                       + len(config.static_variables) + self.space_dim + self.time_dim)

        if self.noise_dim > 0:
            self.noise_generator = NoiseGenerator(self.noise_dim, seed)

        self.preprocess = nn.Sequential(
            nn.Linear(feature_dim, initial_dim, bias=False),
            RMSNorm(initial_dim, elementwise_affine=config.rmsnorm_elementwise_affine),
            nn.SiLU(),
            nn.Linear(initial_dim, initial_dim, bias=False),
            RMSNorm(initial_dim, elementwise_affine=config.rmsnorm_elementwise_affine),
        )

        self.interp_to_hp = CrossAttentionInterpolate(config)
        self.interp_to_ll = CrossAttentionInterpolate(config)

        self.encoder_stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        all_stages = [*config.cg_stage_cfgs, config.bottleneck_cfg]

        for i in range(len(config.cg_stage_cfgs)):
            current_stage = all_stages[i]
            next_stage = all_stages[i + 1]

            self.encoder_stages.append(UNetStage(config=config, stage_cfg=current_stage, depth=current_stage.encoder_depth))
            self.downsample_layers.append(
                HEALPixDownsample(
                    in_dim=current_stage.dim,
                    out_dim=next_stage.dim,
                    nside_before=current_stage.nside,
                    nside_after=next_stage.nside,
                    rmsnorm_elementwise_affine=config.rmsnorm_elementwise_affine,
                )
            )

        self.bottleneck = UNetStage(config=config, stage_cfg=config.bottleneck_cfg, depth=config.bottleneck_cfg.depth)

        self.decoder_stages = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        for i in reversed(range(len(config.cg_stage_cfgs))):
            prev_stage = all_stages[i + 1]
            current_stage = all_stages[i]

            self.upsample_layers.append(
                HEALPixUpsample(
                    in_dim=prev_stage.dim,
                    out_dim=current_stage.dim,
                    nside_before=prev_stage.nside,
                    nside_after=current_stage.nside,
                    rmsnorm_elementwise_affine=config.rmsnorm_elementwise_affine,
                )
            )
            self.decoder_stages.append(UNetStage(config=config, stage_cfg=current_stage, depth=current_stage.decoder_depth))

        self.norm_before_interp_ll = RMSNorm(initial_dim, elementwise_affine=config.rmsnorm_elementwise_affine)

        self.postprocess = nn.Sequential(
            RMSNorm(initial_dim, elementwise_affine=config.rmsnorm_elementwise_affine),
            nn.Linear(initial_dim, initial_dim, bias=False),
            nn.SiLU(),
            nn.Linear(initial_dim, len(config.variables), bias=False),
        )

        self.apply(self._initialize_weights)
        self._zero_init_residual_layers()
        self.initialize_rope()

    def _initialize_weights(self, module):
        if module is self:
            return
        ortho_init = self.config.ortho_init

        if isinstance(module, nn.Linear):
            fan_in, fan_out = module.weight.size(1), module.weight.size(0)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            if ortho_init:
                nn.init.orthogonal_(module.weight); module.weight.data.mul_(std)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None: nn.init.zeros_(module.bias)

    def _zero_init_residual_layers(self):
        ortho_init = self.config.ortho_init

        for stage in [*self.encoder_stages, self.bottleneck, *self.decoder_stages]:
            for block in stage.blocks:
                if ortho_init:
                    nn.init.orthogonal_(block.attention.to_o.weight)
                    block.attention.to_o.weight.data.mul_(0.01)
                    nn.init.orthogonal_(block.ffn.w2.weight)
                    block.ffn.w2.weight.data.mul_(0.01)
                else:
                    nn.init.normal_(block.attention.to_o.weight, mean=0.0, std=0.01)
                    nn.init.normal_(block.ffn.w2.weight, mean=0.0, std=0.01)

                if self.noise_dim > 0:
                    nn.init.normal_(block.ffn.noise_bias.weight, mean=0.0, std=0.01)

        for upsample in self.upsample_layers:
            if ortho_init:
                nn.init.orthogonal_(upsample.proj_x.weight); upsample.proj_x.weight.data.mul_(0.01)
                nn.init.orthogonal_(upsample.proj_pos.weight); upsample.proj_pos.weight.data.mul_(0.01)
            else:
                nn.init.normal_(upsample.proj_x.weight, mean=0.0, std=0.01)
                nn.init.normal_(upsample.proj_pos.weight, mean=0.0, std=0.01)

        if self.noise_dim > 0:
            nn.init.normal_(self.noise_generator.to_noise.weight, mean=0.0, std=0.01)

    def initialize_rope(self):
        if not self.config.rope:
            return
        for stage in [*self.encoder_stages, self.bottleneck, *self.decoder_stages]:
            hp_grid = get_healpix_grid(stage.nside)
            for block in stage.blocks:
                if block.attention.q_rope is not None:
                    block.attention.q_rope.initialize_rope(hp_grid)
                    block.attention.k_rope.initialize_rope(hp_grid)

    def initialize_interpolation(self, longitude: torch.Tensor, latitude: torch.Tensor):
        ll_grid_rad = torch.deg2rad(torch.stack(torch.meshgrid(longitude, latitude, indexing='ij'), -1).reshape(-1, 2))
        hp_grid_rad = torch.deg2rad(get_healpix_grid(self.nside)).to(longitude.device)
        self.interp_to_hp.initialize_interpolation_scheme(ll_grid_rad, hp_grid_rad)
        self.interp_to_ll.initialize_interpolation_scheme(hp_grid_rad, ll_grid_rad)

    @torch.no_grad()
    def initialize_static_vars(self, static_vars: torch.Tensor, longitude: torch.Tensor, latitude: torch.Tensor):
        ll_grid_rad = torch.deg2rad(torch.stack(torch.meshgrid(longitude, latitude, indexing='ij'), -1))
        ll_grid_xyz = rad_to_xyz(ll_grid_rad)
        static_vars = torch.concat([static_vars, ll_grid_xyz], dim=-1)
        static_vars_mean = static_vars.mean(dim=(0, 1), keepdim=True)
        static_vars_std = static_vars.std(dim=(0, 1), keepdim=True) + 1e-6
        static_vars_norm = (static_vars - static_vars_mean) / static_vars_std
        static_vars = rearrange(static_vars_norm, 'lon lat c -> (lon lat) 1 c').contiguous()
        self.register_buffer('static_vars', static_vars, persistent=True)

    @torch.no_grad()
    def time_embedding(self, day_year_time: torch.Tensor):
        day = day_year_time[:, 0:1]
        year = day_year_time[:, 1:2]
        day_sin = torch.sin(2 * math.pi * day)
        day_cos = torch.cos(2 * math.pi * day)
        year_sin = torch.sin(2 * math.pi * year)
        year_cos = torch.cos(2 * math.pi * year)
        return torch.cat([day_sin, day_cos, year_sin, year_cos], dim=-1)

    def forward(self, x: torch.Tensor, day_year_time: torch.Tensor, num_noise_samples: int):
        b, n, _, lon, lat, _ = x.shape
        batch_size = b * num_noise_samples * n

        if self.noise_dim > 0:
            z = self.noise_generator(batch_size, x.device, x.dtype)
        else:
            z = None

        x = repeat(x, 'b n t lon lat c -> (lon lat) (b s n) (t c)', s=num_noise_samples)
        day_year_time = repeat(day_year_time, 'b n d -> (b s n) d', s=num_noise_samples)

        x = torch.cat([
            x,
            self.static_vars.expand(-1, batch_size, -1),
            self.time_embedding(day_year_time).unsqueeze(0).expand(x.shape[0], -1, -1)
        ], dim=-1)

        x = self.preprocess(x)
        x = self.interp_to_hp(x)

        skip_connections = []
        for encoder_stage, downsample in zip(self.encoder_stages, self.downsample_layers):
            x = encoder_stage(x, z)
            skip_connections.append(x)
            x = downsample(x)

        x = self.bottleneck(x, z)

        for decoder_stage, upsample, skip in zip(self.decoder_stages, self.upsample_layers, reversed(skip_connections)):
            x = upsample(x, skip)
            x = decoder_stage(x, z)

        x = self.norm_before_interp_ll(x)
        x = self.interp_to_ll(x)
        x = self.postprocess(x)

        x = rearrange(x, '(lon lat) (b n s) c -> b (n s) lon lat c', lon=lon, lat=lat, b=b, s=num_noise_samples)
        return x
