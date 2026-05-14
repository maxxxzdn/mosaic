"""
Primitive building blocks for the Mosaic transformer.

Components:
- Block-sparse attention with learned strategy weighting (local block, compressed,
  and top-k selection branches combined with a learned gate)
- Rotary positional embeddings (RoPE) for 2D lon/lat
- Cross-attention interpolation between grids
- HEALPix spatial up/downsampling
- Conditional SwiGLU FFN with noise injection
"""

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from torch.nn import RMSNorm

try:
    from flash_attn import flash_attn_func  # FlashAttention v2
except ImportError:
    import flash_attn_interface as fa       # FlashAttention v3
    flash_attn_func = fa.flash_attn_func

from utils import get_healpix_grid, get_neighbors, rad_to_xyz
from ops import mosaic_sparse_attn


def block_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int):
    batch_size = q.shape[0]
    q, k, v = map(lambda x: rearrange(x, 'b (nb bs) h d -> (b nb) bs h d', bs=block_size), (q, k, v))
    o_ba = flash_attn_func(q, k, v)
    return rearrange(o_ba, '(b nb) bs h d -> b (nb bs) h d', b=batch_size)


@torch.no_grad()
def attn_topk(q: torch.Tensor, k: torch.Tensor, block_count: int):
    Hq, Hk = q.shape[2], k.shape[2]
    G = Hq // Hk
    k = k.repeat_interleave(G, dim=2)

    scores = torch.matmul(
        rearrange(q, 'b t h d -> b h t d'),
        rearrange(k, 'b t h d -> b h d t')
    )

    if Hq != Hk:
        scores = reduce(scores, 'b (g h) t k -> b h t k', 'mean', g=G)

    scores = rearrange(scores, 'b h t k -> b t h k')
    top_indices = scores.topk(k=block_count, dim=-1, largest=True)[1]
    return top_indices


def mosaic_attn_func(
    q, k, v,
    weight_ba_cmp_slc,
    block_attn_size, sparse_block_size, sparse_block_count,
    block_attn_only, no_compression=False,
):
    o_ba = block_attention(q, k, v, block_attn_size)

    if block_attn_only:
        return o_ba

    q_cmp = reduce(q, 'b (nb bs) h d -> b nb h d', 'mean', bs=sparse_block_size)
    k_cmp = reduce(k, 'b (nb bs) h d -> b nb h d', 'mean', bs=sparse_block_size)

    if no_compression:
        block_indices = attn_topk(q_cmp, k_cmp, sparse_block_count)
        o_slc = mosaic_sparse_attn(q, k, v, block_indices, sparse_block_size)
        w_ba = weight_ba_cmp_slc[0]
        w_slc = weight_ba_cmp_slc[2]
        w_sum = w_ba + w_slc + 1e-8
        return o_ba * (w_ba / w_sum) + o_slc * (w_slc / w_sum)

    v_cmp = reduce(v, 'b (nb bs) h d -> b nb h d', 'mean', bs=sparse_block_size)
    o_cmp = flash_attn_func(q_cmp, k_cmp, v_cmp)
    o_cmp = o_cmp.repeat_interleave(sparse_block_size, dim=1)

    if sparse_block_count == 0:
        w_ba = weight_ba_cmp_slc[0]
        w_cmp = weight_ba_cmp_slc[1]
        w_sum = w_ba + w_cmp + 1e-8
        return o_ba * (w_ba / w_sum) + o_cmp * (w_cmp / w_sum)

    block_indices = attn_topk(q_cmp, k_cmp, sparse_block_count)
    o_slc = mosaic_sparse_attn(q, k, v, block_indices, sparse_block_size)

    return o_ba * weight_ba_cmp_slc[0] + o_cmp * weight_ba_cmp_slc[1] + o_slc * weight_ba_cmp_slc[2]


class cSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, noise_dim: int):
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.act_fn = nn.SiLU()

        if noise_dim > 0:
            self.noise_bias = nn.Linear(noise_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, z: torch.Tensor = None):
        noise = self.noise_bias(z).unsqueeze(0) if z is not None else 0
        x1, x3 = self.w13(x).chunk(2, dim=-1)
        return self.w2(self.act_fn(x1 + noise) * x3)


class RoPE(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.theta = theta

    def initialize_rope(self, positions):
        base_freqs = 1. / (self.theta ** (torch.arange(0, self.dim // 2, 2).float() / (self.dim // 2)))
        lon_pos = torch.deg2rad(positions[:, 0:1])
        lat_pos = torch.deg2rad(positions[:, 1:2])
        lon_freqs = torch.matmul(lon_pos, base_freqs.unsqueeze(0))
        lat_freqs = torch.matmul(lat_pos, base_freqs.unsqueeze(0))
        freqs = torch.cat([lon_freqs, lat_freqs], dim=-1)
        self.register_buffer('cos_freqs', freqs.cos().contiguous(), persistent=True)
        self.register_buffer('sin_freqs', freqs.sin().contiguous(), persistent=True)

    @staticmethod
    def rotate_half(x):
        x = rearrange(x, '... (d r) -> ... d r', r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d r -> ... (d r)')

    def forward(self, x):
        cos = self.cos_freqs.unsqueeze(0).unsqueeze(2).repeat_interleave(2, dim=-1)
        sin = self.sin_freqs.unsqueeze(0).unsqueeze(2).repeat_interleave(2, dim=-1)
        return (x.float() * cos + self.rotate_half(x.float()) * sin).to(x.dtype)


class MosaicAttention(nn.Module):
    def __init__(self, config, block_attn_only: bool, no_compression: bool = False):
        super().__init__()
        self.block_attn_only = block_attn_only
        self.no_compression = no_compression
        self.block_attn_size = config.block_attn_size
        self.sparse_block_size = config.sparse_block_size
        self.sparse_block_count = config.sparse_block_count

        q_heads = config.num_heads
        gqa_ratio = config.gqa_ratio
        dim = config.dim
        qkv_compress_ratio = config.qkv_compress_ratio
        rope = config.rope
        rope_theta = config.rope_theta

        kv_heads = q_heads // gqa_ratio
        head_dim = int(dim // q_heads // qkv_compress_ratio)

        self.q_heads = q_heads
        self.kv_heads = kv_heads

        self.to_q = nn.Linear(dim, q_heads * head_dim, bias=False)
        self.to_k = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.to_v = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.to_o = nn.Linear(q_heads * head_dim, dim, bias=False)

        self.q_rope = RoPE(head_dim, rope_theta) if rope else None
        self.k_rope = RoPE(head_dim, rope_theta) if rope else None

        if block_attn_only:
            self.to_strategy_combine_mlp = None
        else:
            self.to_strategy_combine_mlp = nn.Linear(dim, 3 * q_heads, bias=False)

    def generate_strategy_weights(self, x):
        if self.block_attn_only:
            return [None, None, None]
        strategy_logits = self.to_strategy_combine_mlp(x)
        strategy_logits = rearrange(strategy_logits, 't b (h s) -> s b t h', h=self.q_heads)
        strategy_weights = torch.softmax(strategy_logits.float(), dim=0).type_as(x)
        strategy_weights = strategy_weights.unsqueeze(-1)
        return strategy_weights

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        strategy_weights = self.generate_strategy_weights(x)

        q = rearrange(q, 's b (h d) -> b s h d', h=self.q_heads)
        k = rearrange(k, 's b (h d) -> b s h d', h=self.kv_heads)
        v = rearrange(v, 's b (h d) -> b s h d', h=self.kv_heads)

        if self.q_rope is not None:
            q = self.q_rope(q)
            k = self.k_rope(k)

        output = mosaic_attn_func(
            q=q, k=k, v=v,
            weight_ba_cmp_slc=strategy_weights,
            block_attn_size=self.block_attn_size,
            sparse_block_size=self.sparse_block_size,
            sparse_block_count=self.sparse_block_count,
            block_attn_only=self.block_attn_only,
            no_compression=self.no_compression,
        )

        output = rearrange(output, 'b s h d -> s b (h d)')
        output = self.to_o(output)
        return output


class MosaicBlock(nn.Module):
    def __init__(self, config, block_attn_only: bool, no_compression: bool = False):
        super().__init__()
        dim = config.dim
        noise_dim = config.noise_dim
        mlp_ratio = config.mlp_ratio

        self.attention = MosaicAttention(config, block_attn_only, no_compression)
        self.norm1 = RMSNorm(dim, elementwise_affine=config.rmsnorm_elementwise_affine)
        self.norm2 = RMSNorm(dim, elementwise_affine=config.rmsnorm_elementwise_affine)
        self.ffn = cSwiGLU(dim, int(dim * mlp_ratio), noise_dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor = None):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x), z)
        return x


class CrossAttentionInterpolate(nn.Module):
    space_dim = 3

    def __init__(self, config):
        super().__init__()
        self.k_neighbors = config.k_neighbors

        dim = config.dim
        num_heads = config.num_heads

        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.kv_norm = RMSNorm(dim, elementwise_affine=config.rmsnorm_elementwise_affine)
        self.to_q = nn.Linear(self.space_dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.to_o = nn.Linear(dim, dim, bias=False)

    @torch.no_grad()
    def initialize_interpolation_scheme(self, pos_from_rad, pos_to_rad):
        neighbors_np = get_neighbors(pos_from_rad.cpu().numpy(), pos_to_rad.cpu().numpy(), k=self.k_neighbors)
        neighbors = torch.from_numpy(neighbors_np).long().to(pos_from_rad.device).contiguous()

        pos_to_xyz = rad_to_xyz(pos_to_rad)
        pos_from_xyz = rad_to_xyz(pos_from_rad)

        rel_pos_xyz = (pos_to_xyz.unsqueeze(1) - pos_from_xyz[neighbors]).contiguous()
        norm_rel_pos_xyz = torch.nn.functional.normalize(rel_pos_xyz, dim=-1).contiguous()

        self.register_buffer('neighbors', neighbors, persistent=True)
        self.register_buffer('rel_pos', norm_rel_pos_xyz, persistent=True)

    def forward(self, x_from: torch.Tensor):
        if self.neighbors is None or self.rel_pos is None:
            raise ValueError("Interpolation scheme not initialized.")

        q = self.to_q(self.rel_pos)
        q = rearrange(q, 's k (h d) -> s k 1 h d', h=self.num_heads)

        x = self.kv_norm(x_from)

        kv = self.to_kv(x)
        kv = rearrange(kv, 's b (n h d) -> n s b h d', h=self.num_heads, n=2)

        k, v = kv[:, self.neighbors]

        attn_scores = (q * k).sum(dim=-1, keepdim=True) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=1, dtype=torch.float32).type_as(k)
        out = (attn_weights * v).sum(dim=1)

        out = rearrange(out, 's b h d -> s b (h d)')
        out = self.to_o(out)
        return out


class NoiseGenerator(nn.Module):
    def __init__(self, noise_dim: int, seed: int):
        super().__init__()
        self.seed = seed
        self.to_noise = nn.Linear(noise_dim, noise_dim, bias=False)
        self.generator = None

    def forward(self, num_samples: int, device: torch.device, dtype: torch.dtype):
        if self.generator is None:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(self.seed)

        noise = torch.randn((num_samples, self.to_noise.in_features),
                            generator=self.generator, device=device, dtype=dtype)
        noise = self.to_noise(noise)
        return noise


class HEALPixDownsample(nn.Module):
    space_dim: int = 3

    def __init__(self, in_dim, out_dim, nside_before, nside_after,
                 rmsnorm_elementwise_affine=True):
        super().__init__()
        self.factor = (nside_before // nside_after) ** 2

        self.proj_x = nn.Linear(self.factor * in_dim, out_dim, bias=False)
        self.proj_pos = nn.Linear(self.factor * self.space_dim, out_dim, bias=False)
        self.norm = RMSNorm(out_dim, elementwise_affine=rmsnorm_elementwise_affine)

        hp_grid_fine_xyz = rad_to_xyz(torch.deg2rad(get_healpix_grid(nside_before)))
        hp_grid_coarse_xyz = rad_to_xyz(torch.deg2rad(get_healpix_grid(nside_after)))

        pos = rearrange(hp_grid_fine_xyz, '(n f) d -> n f d', f=self.factor)
        rel_pos = rearrange(pos - hp_grid_coarse_xyz[:, None], 'n f d -> n (f d)')
        rel_pos = (rel_pos - rel_pos.mean(dim=0, keepdim=True)) / (rel_pos.std(dim=0, keepdim=True) + 1e-6)

        self.register_buffer('rel_pos', rel_pos.contiguous(), persistent=True)

    def forward(self, x: torch.Tensor):
        x = rearrange(x, '(n f) b c -> n b (f c)', f=self.factor)
        x = self.proj_x(x) + self.proj_pos(self.rel_pos).unsqueeze(1)
        x = self.norm(x)
        return x


class HEALPixUpsample(nn.Module):
    space_dim: int = 3

    def __init__(self, in_dim, out_dim, nside_before, nside_after,
                 rmsnorm_elementwise_affine=True):
        super().__init__()
        self.factor = (nside_after // nside_before) ** 2

        self.proj_x = nn.Linear(in_dim, out_dim * self.factor, bias=False)
        self.proj_pos = nn.Linear(self.factor * self.space_dim, out_dim * self.factor, bias=False)
        self.norm = RMSNorm(out_dim, elementwise_affine=rmsnorm_elementwise_affine)

        hp_grid_fine_xyz = rad_to_xyz(torch.deg2rad(get_healpix_grid(nside_after)))
        hp_grid_coarse_xyz = rad_to_xyz(torch.deg2rad(get_healpix_grid(nside_before)))

        children_pos_reshaped = rearrange(hp_grid_fine_xyz, '(n f) d -> n f d', f=self.factor)
        rel_pos = rearrange(children_pos_reshaped - hp_grid_coarse_xyz[:, None], 'n f d -> n (f d)')
        rel_pos = (rel_pos - rel_pos.mean(dim=0, keepdim=True)) / (rel_pos.std(dim=0, keepdim=True) + 1e-6)

        self.register_buffer('rel_pos', rel_pos.contiguous(), persistent=True)

    def forward(self, x: torch.Tensor, shortcut: torch.Tensor):
        x = self.proj_x(x) + self.proj_pos(self.rel_pos).unsqueeze(1)
        x = rearrange(x, 'n b (f d) -> (n f) b d', f=self.factor)
        x = x + shortcut
        x = self.norm(x)
        return x
