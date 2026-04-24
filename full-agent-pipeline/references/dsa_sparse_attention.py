# Benchmark baseline — reference implementation embedded in:
#   mlsys26-contest/definitions/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.json
# Definition: dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
#
# This is the EXACT Python code the flashinfer-bench harness times
# our solution against. Reported "speedup" = ref_latency / our_latency.
# Kept simple to define correctness, not optimized.
#
# Extracted verbatim (2026-04-19) so it can be diffed against the
# optimized solution/python/<name> without spelunking the JSON.
#
# SHAPE KEY (all from definitions/.../json; const values inlined):
#   Nt   = num_tokens      (var; 1..8 in dataset — decode-style batching)
#   H    = num_qo_heads = 16                (Q/output heads)
#   Dc   = head_dim_ckv = 512               (latent KV dim, MLA-style "nope" part)
#   Dp   = head_dim_kpe = 64                (positional "pe" part)
#   PS   = page_size = 64                   (tokens per page)
#   K    = topk = 2048                      (selected pages per query token)
#   P    = num_pages       (var; 8462 in the dataset)
#   Nv   = #valid KV tokens per query token (≤ K; padded entries are −1)

import math
import torch


@torch.no_grad()
def run(
    q_nope,          # [Nt, H=16, Dc=512]              bf16   (content Q)
    q_pe,            # [Nt, H=16, Dp=64]               bf16   (positional Q)
    ckv_cache,       # [P, PS=64, Dc=512]              bf16   (paged latent KV)
    kpe_cache,       # [P, PS=64, Dp=64]               bf16   (paged positional K)
    sparse_indices,  # [Nt, K=2048]                    int32  (flat token ids; -1 = pad)
    sm_scale,        # scalar fp32  (softmax scale)
):                   # returns (output [Nt, H=16, Dc=512] bf16, lse [Nt, H=16] fp32)
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape   # Nt, 16, 512
    head_dim_kpe = q_pe.shape[-1]                           # 64
    num_pages, page_size, _ = ckv_cache.shape               # P, 64, 512
    topk = sparse_indices.shape[-1]                         # 2048

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048

    # Check constraints
    assert sparse_indices.shape[0] == num_tokens
    assert sparse_indices.shape[-1] == topk
    assert ckv_cache.shape[1] == page_size

    device = q_nope.device

    # ────────────────────────────────────────────────────────────────────────
    # Flatten the paged KV so `sparse_indices` can address KV tokens directly
    # as flat offsets in [0, P * PS).
    # ────────────────────────────────────────────────────────────────────────
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [P*PS, Dc=512]  fp32
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [P*PS, Dp=64]   fp32

    output = torch.zeros(
        (num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )                                                       # [Nt, H=16, Dc=512]    bf16
    lse = torch.full(
        (num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )                                                       # [Nt, H=16]            fp32

    # ────────────────────────────────────────────────────────────────────────
    # Per-query-token sparse attention
    # ────────────────────────────────────────────────────────────────────────
    for t in range(num_tokens):
        indices = sparse_indices[t]                         # [K=2048]              int32

        # Handle padding: -1 indicates invalid indices
        valid_mask = indices != -1                          # [K=2048]              bool
        valid_indices = indices[valid_mask]                 # [Nv ≤ K]              int32

        if valid_indices.numel() == 0:
            output[t].zero_()
            continue

        tok_idx = valid_indices.to(torch.long)              # [Nv]                  int64

        Kc = Kc_all[tok_idx]                                # [Nv, Dc=512]          fp32
        Kp = Kp_all[tok_idx]                                # [Nv, Dp=64]           fp32
        qn = q_nope[t].to(torch.float32)                    # [H=16, Dc=512]        fp32
        qp = q_pe[t].to(torch.float32)                      # [H=16, Dp=64]         fp32

        # Attention logits: nope (q_nope · Kc) + pe (q_pe · Kp)
        logits = (qn @ Kc.T) + (qp @ Kp.T)                  # [H,Dc]·[Dc,Nv] + [H,Dp]·[Dp,Nv] = [H=16, Nv]  fp32
        logits_scaled = logits * sm_scale                   # [H=16, Nv]            fp32

        # log-sum-exp in base-2 (for return)
        lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)   # [H=16]  fp32

        # Softmax attention, then attend to Kc (value = the content portion)
        attn = torch.softmax(logits_scaled, dim=-1)         # [H=16, Nv]            fp32
        out = attn @ Kc                                     # [H,Nv] @ [Nv,Dc] = [H=16, Dc=512]    fp32
        output[t] = out.to(torch.bfloat16)                  # [H=16, Dc=512]        bf16

    return output, lse
