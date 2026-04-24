import torch
import triton
import triton.language as tl


# The deep_gemm FP8 KV cache packing:
#   per-page bytes [0 : PS*D)            = fp8 data, shape [PS, D] row-major
#   per-page bytes [PS*D : PS*D + PS*4)  = fp32 scale, one per token
# Per page = PS*D + PS*4 = 64*128 + 64*4 = 8448 bytes (= PS * DSF = 64 * 132).
#
# Strategy
# --------
# The harness checks topk_indices elementwise with required_matched_ratio=1.0.
# To stay aligned with reference top-k tie and NaN behavior, the final
# ReLU/weight/reduce/topk path is run per batch on exactly [H, seq_len].
# The expensive part still avoids the reference's full-cache dequantization:
# gather/dequantize only block_table pages, batch the Q @ K.T matmul, and fuse
# local-to-global index conversion with -1 tail padding.


@triton.jit
def _dequant_gather_kernel(
    K_ptr,          # uint8*       : flat byte view of packed cache
    S_ptr,          # fp32*        : same storage reinterpreted as fp32
    BT_ptr,         # int32*       : [B, Pmax] block_table
    SEQ_ptr,        # int32*       : [B] seq_lens
    OUT_ptr,        # fp32*        : [B, Pmax * PS, D]
    Pmax,
    D: tl.constexpr,
    PS: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    PAGE_F32: tl.constexpr,
    SCALE_F32_OFFSET: tl.constexpr,
):
    b = tl.program_id(0)
    page_tile = tl.program_id(1)

    seq_len = tl.load(SEQ_ptr + b)
    token_start = page_tile * PS

    d_range = tl.arange(0, D)
    p_range = tl.arange(0, PS)

    out_base = (b * Pmax * PS + token_start) * D
    out_off = out_base + p_range[:, None] * D + d_range[None, :]

    if token_start >= seq_len:
        tl.store(OUT_ptr + out_off, tl.zeros([PS, D], dtype=tl.float32))
        return

    global_page = tl.load(BT_ptr + b * Pmax + page_tile)
    gp = global_page.to(tl.int64)

    k_byte_off = gp * PAGE_BYTES + p_range[:, None] * D + d_range[None, :]
    k_bytes = tl.load(K_ptr + k_byte_off)
    k_fp8 = k_bytes.to(tl.float8e4nv, bitcast=True)
    k_fp32 = k_fp8.to(tl.float32)

    s_off = gp * PAGE_F32 + SCALE_F32_OFFSET + p_range
    scale = tl.load(S_ptr + s_off)

    tl.store(OUT_ptr + out_off, k_fp32 * scale[:, None])


@triton.jit
def _convert_indices_kernel(
    LOC_ptr,        # int64*: [actual_topk] per-batch local topk indices
    BT_ptr,         # int32*: [Pmax]        per-batch block_table row
    OUT_ptr,        # int32*: [topk]        per-batch destination row
    actual_topk,
    topk: tl.constexpr,
    PS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    mask_total = offs < topk
    mask_valid = offs < actual_topk

    loc = tl.load(LOC_ptr + offs, mask=mask_valid, other=0)
    page_slot = loc // PS
    off_in_page = loc % PS

    bt_val = tl.load(BT_ptr + page_slot, mask=mask_valid, other=0)
    global_tok = bt_val.to(tl.int64) * PS + off_in_page
    result = tl.where(mask_valid, global_tok.to(tl.int32), -1)

    tl.store(OUT_ptr + offs, result, mask=mask_total)


@torch.no_grad()
def run(
    q_index_fp8,
    k_index_cache_fp8,
    weights,
    seq_lens,
    block_table,
    topk_indices,
):
    B, H, D = q_index_fp8.shape
    P, PS, _, DSF = k_index_cache_fp8.shape
    topk = 2048
    Pmax = block_table.shape[1]
    device = q_index_fp8.device

    PAGE_BYTES = PS * DSF
    PAGE_F32 = PAGE_BYTES // 4
    SCALE_F32_OFFSET = (PS * D) // 4
    max_tokens = Pmax * PS

    if B == 0:
        return
    if max_tokens == 0:
        topk_indices.fill_(-1)
        return

    q_c = q_index_fp8.contiguous()
    k_c = k_index_cache_fp8.contiguous()
    w_c = weights.contiguous()
    bt_c = block_table.contiguous()

    k_bytes = k_c.view(torch.uint8)
    k_flat_fp32 = k_bytes.reshape(-1).view(torch.float32)

    K_dq = torch.empty((B, max_tokens, D), dtype=torch.float32, device=device)
    _dequant_gather_kernel[(B, Pmax)](
        k_bytes, k_flat_fp32, bt_c, seq_lens, K_dq,
        Pmax,
        D=D, PS=PS,
        PAGE_BYTES=PAGE_BYTES,
        PAGE_F32=PAGE_F32,
        SCALE_F32_OFFSET=SCALE_F32_OFFSET,
        num_warps=4,
        num_stages=1,
    )

    q_fp32 = q_c.to(torch.float32)
    scores_hm = torch.matmul(q_fp32, K_dq.transpose(-1, -2))

    seq_lens_cpu = seq_lens.tolist()
    POST_BLOCK = 256
    post_grid = (triton.cdiv(topk, POST_BLOCK),)

    for b in range(B):
        seq_len = seq_lens_cpu[b]
        if seq_len == 0:
            topk_indices[b].fill_(-1)
            continue

        sc = scores_hm[b, :, :seq_len]
        sc = torch.relu(sc)
        sc = sc * w_c[b][:, None]
        final_scores = sc.sum(dim=0)

        actual_topk = topk if seq_len > topk else seq_len
        _, topk_local_idx = torch.topk(final_scores, actual_topk)

        _convert_indices_kernel[post_grid](
            topk_local_idx, bt_c[b], topk_indices[b],
            actual_topk,
            topk=topk, PS=PS, BLOCK=POST_BLOCK,
            num_warps=2,
        )
