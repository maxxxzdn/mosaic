import torch
import triton
import triton.language as tl



def get_autotuning_configs(q_tile_sizes: list):
    """Generate autotuning configurations optimized for H100."""
    warps = [4, 8]
    stages = [2, 3]

    return [
        triton.Config({'q_tile_size': t}, num_warps=w, num_stages=s)
        for t in q_tile_sizes
        for w in warps
        for s in stages
    ]


@triton.autotune(
    configs=get_autotuning_configs([64, 128]),
    key=['seq_len', 'feature_dim'],
)
@triton.jit
def mosaic_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr, lse_ptr, block_indices_ptr,
    softmax_scale: tl.constexpr,
    seq_len: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_q_heads: tl.constexpr,
    q_heads_per_kv_head: tl.constexpr,
    feature_dim: tl.constexpr,
    kv_block_size: tl.constexpr,
    num_kv_blocks_per_q_block: tl.constexpr,
    q_tile_size: tl.constexpr,
):
    """
    Sparse attention forward kernel:
        for each query tile (i.e. block chunk), for each query head, attend to a subset of key/value blocks.
    """
    LOG2_E: tl.constexpr = 1.44269504089

    q_tile_id = tl.program_id(0)
    q_head_id = tl.program_id(1)
    batch_kv_head_id = tl.program_id(2)

    batch_idx = batch_kv_head_id // num_kv_heads
    kv_head_idx = batch_kv_head_id % num_kv_heads
    q_head_idx = kv_head_idx * q_heads_per_kv_head + q_head_id

    batch_offset = batch_idx * seq_len
    q_tile_start = q_tile_id * q_tile_size
    num_blocks_in_seq = seq_len // kv_block_size
    tiles_per_block = kv_block_size // q_tile_size
    q_block_id = q_tile_id // tiles_per_block

    block_indices_offset = (
        batch_idx * num_blocks_in_seq * num_kv_heads * num_kv_blocks_per_q_block +
        q_block_id * num_kv_heads * num_kv_blocks_per_q_block +
        kv_head_idx * num_kv_blocks_per_q_block
    )

    q_base_ptr = q_ptr + batch_offset * num_q_heads  * feature_dim + q_head_idx  * feature_dim
    k_base_ptr = k_ptr + batch_offset * num_kv_heads * feature_dim + kv_head_idx * feature_dim
    v_base_ptr = v_ptr + batch_offset * num_kv_heads * feature_dim + kv_head_idx * feature_dim

    q_tile_ptr = tl.make_block_ptr(
        base=q_base_ptr,
        shape=(seq_len, feature_dim),
        strides=(num_q_heads * feature_dim, 1),
        offsets=(q_tile_start, 0),
        block_shape=(q_tile_size, feature_dim),
        order=(1, 0)
    )

    output_tile_ptr = tl.make_block_ptr(
        base=output_ptr + batch_offset * num_q_heads * feature_dim + q_head_idx * feature_dim,
        shape=(seq_len, feature_dim),
        strides=(num_q_heads * feature_dim, 1),
        offsets=(q_tile_start, 0),
        block_shape=(q_tile_size, feature_dim),
        order=(1, 0)
    )

    lse_base_ptr = lse_ptr + (batch_offset + q_tile_start) * num_q_heads + tl.arange(0, q_tile_size) * num_q_heads + q_head_idx

    output_accum = tl.zeros([q_tile_size, feature_dim], dtype=tl.float32)
    max_scores = tl.full([q_tile_size], float('-inf'), dtype=tl.float32)
    sum_exp_scores = tl.zeros([q_tile_size], dtype=tl.float32)

    q_tile = tl.load(q_tile_ptr)
    q_tile = (q_tile * softmax_scale * LOG2_E).to(tl.bfloat16)

    for i in range(num_kv_blocks_per_q_block):
        kv_block_start = kv_block_size * tl.load(block_indices_ptr + block_indices_offset + i).to(tl.int32)

        k_block_ptr = tl.make_block_ptr(
            base=k_base_ptr,
            shape=(feature_dim, seq_len),
            strides=(1, num_kv_heads * feature_dim),
            offsets=(0, kv_block_start),
            block_shape=(feature_dim, kv_block_size),
            order=(1, 0)
        )

        v_block_ptr = tl.make_block_ptr(
            base=v_base_ptr,
            shape=(seq_len, feature_dim),
            strides=(num_kv_heads * feature_dim, 1),
            offsets=(kv_block_start, 0),
            block_shape=(kv_block_size, feature_dim),
            order=(1, 0)
        )

        k_block = tl.load(k_block_ptr).to(tl.bfloat16)
        v_block = tl.load(v_block_ptr).to(tl.bfloat16)

        attention_scores = tl.dot(q_tile, k_block)

        new_max = tl.max(attention_scores, axis=1)
        old_max = max_scores
        max_scores = tl.maximum(max_scores, new_max)
        rescale = tl.exp2(old_max - max_scores)
        attention_probs = tl.exp2(attention_scores - max_scores[:, None])
        sum_exp_scores = sum_exp_scores * rescale + tl.sum(attention_probs, axis=1)

        output_accum = output_accum * rescale[:, None]
        output_accum += tl.dot(attention_probs.to(tl.bfloat16), v_block)

    final_output = output_accum / sum_exp_scores[:, None]
    log_sum_exp = (max_scores + tl.log2(sum_exp_scores))

    tl.store(output_tile_ptr, final_output.to(q_ptr.dtype.element_ty))
    tl.store(lse_base_ptr, log_sum_exp.to(tl.float32))


def mosaic_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_size: int,
    softmax_scale: float,
):
    batch_size, seq_len, num_kv_heads, feature_dim = k.shape
    num_q_heads = q.shape[2]
    num_kv_blocks_per_q_block = block_indices.shape[-1]
    q_heads_per_kv_head = num_q_heads // num_kv_heads

    output = torch.empty(batch_size, seq_len, num_q_heads, feature_dim, dtype=v.dtype, device=q.device)
    lse = torch.empty(batch_size, seq_len, num_q_heads, dtype=torch.float32, device=q.device)

    grid = lambda META: (
        triton.cdiv(seq_len, META['q_tile_size']),
        q_heads_per_kv_head,
        batch_size * num_kv_heads
    )

    mosaic_attn_fwd_kernel[grid](
        q_ptr = q,
        k_ptr = k,
        v_ptr = v,
        output_ptr = output,
        lse_ptr = lse,
        block_indices_ptr = block_indices,
        softmax_scale = softmax_scale,
        seq_len = seq_len,
        num_kv_heads = num_kv_heads,
        num_q_heads = num_q_heads,
        q_heads_per_kv_head = q_heads_per_kv_head,
        feature_dim = feature_dim,
        kv_block_size = block_size,
        num_kv_blocks_per_q_block = num_kv_blocks_per_q_block,
    )

    return output, lse


@triton.autotune(
    configs=get_autotuning_configs([64, 128]),
    key=['seq_len', 'feature_dim'],
)
@triton.jit
def mosaic_attn_bwd_q_kernel(
    q_ptr, k_ptr, v_ptr, lse_ptr, delta_ptr, grad_o_ptr, grad_q_ptr, block_indices_ptr,
    softmax_scale: tl.constexpr,
    seq_len: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_q_heads: tl.constexpr,
    q_heads_per_kv_head: tl.constexpr,
    feature_dim: tl.constexpr,
    kv_block_size: tl.constexpr,
    num_kv_blocks_per_q_block: tl.constexpr,
    q_tile_size: tl.constexpr,
):
    LOG2_E: tl.constexpr = 1.44269504089
    LN_2: tl.constexpr = 0.69314718056

    q_tile_id = tl.program_id(0)
    q_head_id = tl.program_id(1)
    batch_kv_head_id = tl.program_id(2)

    batch_idx = batch_kv_head_id // num_kv_heads
    kv_head_idx = batch_kv_head_id % num_kv_heads
    q_head_idx = kv_head_idx * q_heads_per_kv_head + q_head_id

    batch_offset = batch_idx * seq_len
    q_tile_start = q_tile_id * q_tile_size
    tiles_per_block = kv_block_size // q_tile_size
    q_block_id = q_tile_id // tiles_per_block
    num_q_blocks = seq_len // kv_block_size

    block_indices_offset = (
        batch_idx * num_q_blocks * num_kv_heads * num_kv_blocks_per_q_block +
        q_block_id * num_kv_heads * num_kv_blocks_per_q_block +
        kv_head_idx * num_kv_blocks_per_q_block
    )

    q_offsets = (
        tl.arange(0, q_tile_size)[:, None] * num_q_heads * feature_dim +
        q_head_idx * feature_dim +
        tl.arange(0, feature_dim)[None, :]
    )

    lse_offsets = tl.arange(0, q_tile_size) * num_q_heads + q_head_idx

    q_base_ptr = q_ptr + (batch_offset + q_tile_start) * num_q_heads * feature_dim
    grad_o_base_ptr = grad_o_ptr + (batch_offset + q_tile_start) * num_q_heads * feature_dim
    delta_base_ptr = delta_ptr + (batch_offset + q_tile_start) * num_q_heads
    lse_base_ptr = lse_ptr + (batch_offset + q_tile_start) * num_q_heads
    grad_q_base_ptr = grad_q_ptr + (batch_offset + q_tile_start) * num_q_heads * feature_dim

    grad_q_accum = tl.zeros([q_tile_size, feature_dim], dtype=tl.float32)

    q_tile = tl.load(q_base_ptr + q_offsets)
    q_tile = (q_tile * softmax_scale * LOG2_E).to(tl.bfloat16)

    grad_o_tile = tl.load(grad_o_base_ptr + q_offsets).to(tl.bfloat16)
    delta_vals = tl.load(delta_base_ptr + lse_offsets)
    lse_vals = tl.load(lse_base_ptr + lse_offsets).to(tl.float32)

    for i in range(num_kv_blocks_per_q_block):
        kv_block_idx = tl.load(block_indices_ptr + block_indices_offset + i).to(tl.int32)

        k_block_ptr = tl.make_block_ptr(
            base=k_ptr + (batch_offset * num_kv_heads + kv_head_idx) * feature_dim,
            shape=(feature_dim, seq_len),
            strides=(1, num_kv_heads * feature_dim),
            offsets=(0, kv_block_idx * kv_block_size),
            block_shape=(feature_dim, kv_block_size),
            order=(0, 1)
        )

        v_block_ptr = tl.make_block_ptr(
            base=v_ptr + (batch_offset * num_kv_heads + kv_head_idx) * feature_dim,
            shape=(feature_dim, seq_len),
            strides=(1, num_kv_heads * feature_dim),
            offsets=(0, kv_block_idx * kv_block_size),
            block_shape=(feature_dim, kv_block_size),
            order=(0, 1)
        )

        k_block = tl.load(k_block_ptr).to(tl.bfloat16)
        v_block = tl.load(v_block_ptr).to(tl.bfloat16)

        attention_scores = tl.dot(q_tile, k_block)
        attention_probs = tl.exp2(attention_scores - lse_vals[:, None]) * LN_2

        grad_times_v = tl.dot(grad_o_tile, v_block)
        grad_scores = attention_probs * (grad_times_v - delta_vals[:, None])
        grad_q_accum += tl.dot(grad_scores.to(tl.bfloat16), tl.trans(k_block.to(tl.bfloat16)))

    grad_q_accum = grad_q_accum * softmax_scale * LOG2_E
    tl.store(grad_q_base_ptr + q_offsets, grad_q_accum.to(q_ptr.dtype.element_ty))


@torch.compile
@torch.no_grad()
def mosaic_block_mask(
    block_indices: torch.LongTensor,
):
    batch_size, num_blocks, num_heads, _ = block_indices.shape

    block_mask = torch.zeros(
        batch_size, num_blocks, num_heads, num_blocks,
        dtype=torch.bool, device=block_indices.device
    )

    batch_idx = torch.arange(batch_size, device=block_indices.device)[:, None, None, None]
    q_block_idx = torch.arange(num_blocks, device=block_indices.device)[None, :, None, None]
    head_idx = torch.arange(num_heads, device=block_indices.device)[None, None, :, None]

    block_mask[batch_idx, q_block_idx, head_idx, block_indices] = True

    block_mask_transposed = block_mask.permute(0, 2, 3, 1).contiguous()

    return block_mask_transposed


@triton.autotune(
    configs=get_autotuning_configs([16, 32]),
    key=['seq_len', 'feature_dim'],
)
@triton.jit
def mosaic_attn_bwd_kv_kernel(
    q_ptr, k_ptr, v_ptr, lse_ptr, delta_ptr,
    grad_o_ptr, grad_k_ptr, grad_v_ptr,
    block_mask_ptr,
    softmax_scale: tl.constexpr,
    seq_len: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_q_heads: tl.constexpr,
    q_heads_per_kv_head: tl.constexpr,
    feature_dim: tl.constexpr,
    kv_block_size: tl.constexpr,
    q_tile_size: tl.constexpr,
):
    LOG2_E: tl.constexpr = 1.44269504089
    LN_2: tl.constexpr = 0.69314718056

    kv_block_id = tl.program_id(0)
    batch_kv_head_id = tl.program_id(1)

    batch_idx = batch_kv_head_id // num_kv_heads
    kv_head_idx = batch_kv_head_id % num_kv_heads
    batch_offset = batch_idx * seq_len

    num_blocks_in_seq = seq_len // kv_block_size
    tiles_per_block = kv_block_size // q_tile_size

    fine_mask_start = (
        batch_idx * num_kv_heads * num_blocks_in_seq * num_blocks_in_seq +
        kv_head_idx * num_blocks_in_seq * num_blocks_in_seq +
        kv_block_id * num_blocks_in_seq
    )

    k_block_ptr = tl.make_block_ptr(
        k_ptr + (batch_offset * num_kv_heads + kv_head_idx) * feature_dim,
        (seq_len, feature_dim), (num_kv_heads * feature_dim, 1),
        (kv_block_id * kv_block_size, 0), (kv_block_size, feature_dim), (1, 0)
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + (batch_offset * num_kv_heads + kv_head_idx) * feature_dim,
        (seq_len, feature_dim), (num_kv_heads * feature_dim, 1),
        (kv_block_id * kv_block_size, 0), (kv_block_size, feature_dim), (1, 0)
    )

    grad_k_ptr = tl.make_block_ptr(
        grad_k_ptr + (batch_offset * num_kv_heads + kv_head_idx) * feature_dim,
        (seq_len, feature_dim), (num_kv_heads * feature_dim, 1),
        (kv_block_id * kv_block_size, 0), (kv_block_size, feature_dim), (1, 0)
    )

    grad_v_ptr = tl.make_block_ptr(
        grad_v_ptr + (batch_offset * num_kv_heads + kv_head_idx) * feature_dim,
        (seq_len, feature_dim), (num_kv_heads * feature_dim, 1),
        (kv_block_id * kv_block_size, 0), (kv_block_size, feature_dim), (1, 0)
    )

    k_block = tl.load(k_block_ptr).to(tl.bfloat16)
    v_block = tl.load(v_block_ptr).to(tl.bfloat16)

    grad_k_accum = tl.zeros([kv_block_size, feature_dim], dtype=tl.float32)
    grad_v_accum = tl.zeros([kv_block_size, feature_dim], dtype=tl.float32)

    for q_block_id in range(num_blocks_in_seq):
        is_connected = tl.load(block_mask_ptr + fine_mask_start + q_block_id)

        if is_connected:
            for tile_in_block in range(tiles_per_block):
                tile_idx = q_block_id * tiles_per_block + tile_in_block
                q_tile_start = tile_idx * q_tile_size

                q_tile_ptr = tl.make_block_ptr(
                    base=q_ptr + (batch_offset + q_tile_start) * num_q_heads * feature_dim,
                    shape=(q_tile_size, num_q_heads, feature_dim),
                    strides=(num_q_heads * feature_dim, feature_dim, 1),
                    offsets=(0, kv_head_idx * q_heads_per_kv_head, 0),
                    block_shape=(q_tile_size, q_heads_per_kv_head, feature_dim),
                    order=(0, 1, 2),
                )

                grad_o_tile_ptr = tl.make_block_ptr(
                    base=grad_o_ptr + (batch_offset + q_tile_start) * num_q_heads * feature_dim,
                    shape=(q_tile_size, num_q_heads, feature_dim),
                    strides=(num_q_heads * feature_dim, feature_dim, 1),
                    offsets=(0, kv_head_idx * q_heads_per_kv_head, 0),
                    block_shape=(q_tile_size, q_heads_per_kv_head, feature_dim),
                    order=(0, 1, 2),
                )

                lse_tile_ptr = tl.make_block_ptr(
                    base=lse_ptr + (batch_offset + q_tile_start) * num_q_heads,
                    shape=(q_tile_size, num_q_heads),
                    strides=(num_q_heads, 1),
                    offsets=(0, kv_head_idx * q_heads_per_kv_head),
                    block_shape=(q_tile_size, q_heads_per_kv_head),
                    order=(1, 0),
                )

                delta_tile_ptr = tl.make_block_ptr(
                    base=delta_ptr + (batch_offset + q_tile_start) * num_q_heads,
                    shape=(q_tile_size, num_q_heads),
                    strides=(num_q_heads, 1),
                    offsets=(0, kv_head_idx * q_heads_per_kv_head),
                    block_shape=(q_tile_size, q_heads_per_kv_head),
                    order=(1, 0),
                )

                q_tile = tl.load(q_tile_ptr) * softmax_scale * LOG2_E
                q_tile = tl.reshape(q_tile, (q_tile_size * q_heads_per_kv_head, feature_dim))
                q_tile = q_tile.to(tl.bfloat16)

                grad_o_block = tl.load(grad_o_tile_ptr)
                grad_o_block = tl.reshape(grad_o_block, (q_tile_size * q_heads_per_kv_head, feature_dim))
                grad_o_block = grad_o_block.to(tl.bfloat16)

                lse_vals = tl.load(lse_tile_ptr)
                lse_vals = tl.reshape(lse_vals, (q_tile_size * q_heads_per_kv_head,))

                delta_vals = tl.load(delta_tile_ptr)
                delta_vals = tl.reshape(delta_vals, (q_tile_size * q_heads_per_kv_head,))

                attention_scores = tl.dot(k_block, tl.trans(q_tile))
                attention_probs = tl.exp2(attention_scores - lse_vals[None, :])
                grad_v_accum += tl.dot(attention_probs.to(tl.bfloat16), grad_o_block)
                grad_times_v = tl.dot(v_block, tl.trans(grad_o_block))
                grad_scores = attention_probs * (grad_times_v - delta_vals[None, :]) * LN_2
                grad_k_accum += tl.dot(grad_scores.to(tl.bfloat16), q_tile)

    tl.store(grad_k_ptr, grad_k_accum.to(grad_k_ptr.dtype.element_ty))
    tl.store(grad_v_ptr, grad_v_accum.to(grad_v_ptr.dtype.element_ty))


def mosaic_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_o: torch.Tensor,
    softmax_scale: float,
    block_indices: torch.LongTensor,
    block_size: int,
):
    batch_size, seq_len, num_kv_heads, feature_dim = k.shape
    num_q_heads = q.shape[2]
    num_kv_blocks_per_q_block = block_indices.shape[-1]
    q_heads_per_kv_head = num_q_heads // num_kv_heads
    num_blocks_in_seq = seq_len // block_size

    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)

    block_mask = mosaic_block_mask(block_indices)

    delta = (output * grad_o).sum(dim=-1)

    grid_dq = lambda META: (
        triton.cdiv(seq_len, META['q_tile_size']),
        q_heads_per_kv_head,
        batch_size * num_kv_heads
    )

    mosaic_attn_bwd_q_kernel[grid_dq](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        lse_ptr=lse,
        delta_ptr=delta,
        grad_o_ptr=grad_o,
        grad_q_ptr=grad_q,
        block_indices_ptr=block_indices,
        softmax_scale=softmax_scale,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        num_q_heads=num_q_heads,
        q_heads_per_kv_head=q_heads_per_kv_head,
        feature_dim=feature_dim,
        kv_block_size=block_size,
        num_kv_blocks_per_q_block=num_kv_blocks_per_q_block,
    )

    grid_dkv = (num_blocks_in_seq, batch_size * num_kv_heads)

    mosaic_attn_bwd_kv_kernel[grid_dkv](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        lse_ptr=lse,
        delta_ptr=delta,
        grad_o_ptr=grad_o,
        grad_k_ptr=grad_k,
        grad_v_ptr=grad_v,
        block_mask_ptr=block_mask,
        softmax_scale=softmax_scale,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        num_q_heads=num_q_heads,
        q_heads_per_kv_head=q_heads_per_kv_head,
        feature_dim=feature_dim,
        kv_block_size=block_size,
    )

    return grad_q, grad_k, grad_v


class MosaicAttnFunction(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indices: torch.Tensor,
        block_size: int,
        softmax_scale: float
    ):
        q, k, v, block_indices = map(lambda x: x.contiguous(), (q, k, v, block_indices))

        ctx.dtype = q.dtype

        output, lse = mosaic_attn_fwd(
            q=q, k=k, v=v,
            block_indices=block_indices,
            block_size=block_size,
            softmax_scale=softmax_scale,
        )

        ctx.save_for_backward(q, k, v, output, lse, block_indices)
        ctx.block_size = block_size
        ctx.softmax_scale = softmax_scale

        return output.to(q.dtype)

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_o: torch.Tensor
    ):
        q, k, v, output, lse, block_indices = ctx.saved_tensors
        grad_o = grad_o.contiguous()
        grad_q, grad_k, grad_v = mosaic_attn_bwd(
            q=q, k=k, v=v, output=output, lse=lse, grad_o=grad_o,
            softmax_scale=ctx.softmax_scale,
            block_indices=block_indices,
            block_size=ctx.block_size,
        )
        return grad_q, grad_k, grad_v, None, None, None


def mosaic_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_size: int,
    softmax_scale: float = None,
):
    softmax_scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    return MosaicAttnFunction.apply(q, k, v, block_indices, block_size, softmax_scale)
