# Benchmark baseline — reference implementation embedded in:
#   mlsys26-contest/definitions/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json
# Definition: dsa_topk_indexer_fp8_h64_d128_topk2048_ps64
#
# This is the EXACT Python code the flashinfer-bench harness times
# our solution against. Reported "speedup" = ref_latency / our_latency.
# Kept simple to define correctness, not optimized.
#
# Extracted verbatim (2026-04-19) so it can be diffed against the
# optimized solution/python/<name> without spelunking the JSON.
#
# SHAPE KEY (all from definitions/.../json; const values inlined):
#   B     = batch_size         (var; 1..31 in dataset)
#   H     = num_index_heads = 64
#   D     = index_head_dim  = 128
#   PS    = page_size       = 64              (tokens per page)
#   K     = topk            = 2048            (output top-K per batch elem)
#   P     = num_pages       (var; 11923 in dataset)
#   Pmax  = max_num_pages   (var; per-batch column of block_table)
#   Hkv   = kv_cache_num_heads = 1
#   D+SF  = head_dim_with_scale = 132         (= D + 4 bytes scale per token)
#   Ls    = per-batch seq_len in tokens       (variable, ≤ Pmax * PS)
#   Ps    = per-batch #pages = ceil(Ls/PS)

import torch


def dequant_fp8_kv_cache(k_index_cache_fp8):                 # [P, PS=64, Hkv=1, D+SF=132]  int8 (interpret as uint8)
    """Dequantize FP8 KV cache from deep_gemm format.

    Input: [num_pages, page_size, 1, 132] int8 (interpreted as uint8)
           Memory layout (per page): [fp8_data (page_size * 128 bytes), scales (page_size * 4 bytes)]
           After view to [num_pages, page_size, 1, 132]: NOT directly indexable as [fp8, scale] per token!
    Output: [num_pages, page_size, 128] float32
    """
    # View as uint8 for correct byte interpretation
    k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)         # [P, PS=64, 1, 132]    uint8
    num_pages, page_size, num_heads, head_dim_sf = k_index_cache_fp8.shape  # P, 64, 1, 132
    head_dim = head_dim_sf - 4                                # 128

    # Flatten the page so we can slice [fp8 bytes | scale bytes]
    kv_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_sf)    # [P, PS*132 = 8448]  uint8

    # FP8 data: first PS*D bytes of each page
    fp8_bytes = kv_flat[:, :page_size * head_dim].contiguous()      # [P, PS*D = 8192]     uint8
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)  # [P, PS=64, D=128]  fp8_e4m3fn
    fp8_float = fp8_tensor.to(torch.float32)                  # [P, PS=64, D=128]    fp32

    # Scale data: last PS*4 bytes of each page (one fp32 per token)
    scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()    # [P, PS*4 = 256]      uint8
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)   # [P, PS=64, 1]   fp32

    return fp8_float * scale                                  # [P, PS=64, D=128]    fp32


@torch.no_grad()
def run(
    q_index_fp8,         # [B, H=64, D=128]                   fp8_e4m3fn
    k_index_cache_fp8,   # [P, PS=64, Hkv=1, D+SF=132]        int8 (deep-gemm packed)
    weights,             # [B, H=64]                          fp32 (per-batch per-head)
    seq_lens,            # [B]                                int32 (per-batch sequence length)
    block_table,         # [B, Pmax]                          int32 (per-batch page ids)
):                       # returns (topk_indices [B, K=2048] int32,)
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape  # B, 64, 128
    num_pages, page_size, _, _ = k_index_cache_fp8.shape             # P, 64, 1, 132
    topk = 2048

    # Check constants
    assert num_index_heads == 64
    assert index_head_dim == 128
    assert page_size == 64

    device = q_index_fp8.device

    # Dequantize inputs
    q = q_index_fp8.to(torch.float32)                         # [B, H=64, D=128]         fp32
    K_all = dequant_fp8_kv_cache(k_index_cache_fp8)           # [P, PS=64, D=128]        fp32

    topk_indices = torch.full(
        (batch_size, topk), -1, dtype=torch.int32, device=device
    )                                                         # [B, K=2048]              int32 (init=-1)
    max_num_pages = block_table.shape[1]                      # Pmax

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())                     # scalar   Ls

        if seq_len == 0:
            continue

        # Gather only the pages this sequence actually uses
        num_pages_for_seq = (seq_len + page_size - 1) // page_size      # scalar Ps = ⌈Ls/PS⌉
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)  # [Ps]   int64

        K_paged = K_all[page_indices]                         # [Ps, PS=64, D=128]       fp32
        K = K_paged.reshape(-1, index_head_dim)[:seq_len]     # [Ls, D=128]              fp32  (trim padding)

        q_b = q[b]                                            # [H=64, D=128]            fp32

        # Attention scores: all heads × all KV tokens for this batch elem
        scores = q_b @ K.T                                    # [H,D] @ [D,Ls] = [H=64, Ls]     fp32

        # ReLU activation (deep_gemm uses ReLU, not softmax)
        scores_relu = torch.relu(scores)                      # [H=64, Ls]               fp32

        # Per-head learned weight, then sum across heads → one score per KV token
        w = weights[b]                                        # [H=64]                   fp32
        weighted_scores = scores_relu * w[:, None]            # [H=64, Ls]               fp32
        final_scores = weighted_scores.sum(dim=0)             # [Ls]                     fp32

        # Select top-K KV tokens (K capped by actual seq len)
        actual_topk = min(topk, seq_len)                      # scalar
        _, topk_idx = torch.topk(final_scores, actual_topk)   # [actual_topk]            int64  (local ids in [0, Ls))

        # Convert local token ids to GLOBAL flat token ids (page_idx * PS + offset)
        page_idx_per_token = topk_idx // page_size            # [actual_topk]  int64  (into page_indices)
        offset_per_token = topk_idx % page_size               # [actual_topk]  int64
        global_page_idx = page_indices[page_idx_per_token]    # [actual_topk]  int64  (global page ids)
        topk_tokens = global_page_idx * page_size + offset_per_token  # [actual_topk]  int64

        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)   # write into [B, K=2048]

    return (topk_indices,)                                    # [B, K=2048]              int32
