# Benchmark baseline — reference implementation embedded in:
#   mlsys26-contest/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json
# Definition: gdn_prefill_qk4_v8_d128_k_last
#
# This is the EXACT Python code the flashinfer-bench harness times
# our solution against. Reported "speedup" = ref_latency / our_latency.
# Kept simple to define correctness, not optimized.
#
# Extracted verbatim (2026-04-19) so it can be diffed against the
# optimized solution/python/<name> without spelunking the JSON.
#
# SHAPE KEY (all from definitions/.../json; const values inlined):
#   Ttot  = total_seq_len                   (var; sum of seq_len across sequences)
#   N     = num_seqs                        (var; Ttot = sum of per-seq len)
#   Hq    = num_q_heads = 4                 (query / key heads; GVA)
#   Hk    = num_k_heads = 4
#   Hv    = num_v_heads = 8                 (value heads; state is per V head)
#   K     = V = head_size = 128             (same for Q, K, V)
#   Lcs   = len_cu_seqlens = N + 1          (cumulative seqlens array length)
#   H     = num_sab_heads = max(Hq, Hv) = 8 (output uses V-head count)
#   GVA repeat factor: Hv/Hq = Hv/Hk = 2    (each Q/K head shared across 2 V heads)

import math
import torch
import torch.nn.functional as F


def matmul(a: torch.Tensor, b: torch.Tensor):
    """Float32 matmul for numerical stability."""
    return a.float() @ b.float()


@torch.no_grad()
def run(
    q,           # [Ttot, Hq=4, K=128]            bf16
    k,           # [Ttot, Hk=4, K=128]            bf16
    v,           # [Ttot, Hv=8, V=128]            bf16
    state,       # [N, Hv=8, V=128, K=128]        fp32   (K-LAST layout; one per seq)
    A_log,       # [Hv=8]                         fp32   (learned per-head decay)
    a,           # [Ttot, Hv=8]                   bf16   (per-token gate logit)
    dt_bias,     # [Hv=8]                         fp32
    b,           # [Ttot, Hv=8]                   bf16   (per-token mixing logit)
    cu_seqlens,  # [N+1]                          int64  (boundaries: cu_seqlens[i]..cu_seqlens[i+1])
    scale,       # scalar fp32 (if None/0 → 1/√K)
):               # returns (output [Ttot, Hv=8, V=128] bf16, new_state [N, Hv=8, V=128, K=128] fp32)
    """
    Gated Delta Net prefill reference implementation (k-last layout).

    State layout: [H, V, K] (k-last, K dimension at the end)

    Gate computation:
    g = exp(-exp(A_log) * softplus(a + dt_bias))
    beta = sigmoid(b)

    Delta rule update:
    state_new = g * state_old + k^T @ (beta * v + (1-beta) * k @ state_old) - k^T @ (k @ state_old)
    output = scale * q @ state_new
    """
    total_seq_len, num_q_heads, head_size = q.shape              # Ttot, 4, 128
    num_v_heads = v.shape[1]                                     # 8
    num_k_heads = k.shape[1]                                     # 4
    num_sab_heads = max(num_q_heads, num_v_heads)                # 8
    num_seqs = cu_seqlens.size(0) - 1                            # N
    device = q.device

    assert num_q_heads == 4
    assert num_k_heads == 4
    assert num_v_heads == 8
    assert head_size == 128

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)                       # scalar fp32

    # ────────────────────────────────────────────────────────────────────────
    # Gates (g: decay, beta: mixing) — computed ALL-AT-ONCE for all tokens
    # ────────────────────────────────────────────────────────────────────────
    x = a.float() + dt_bias.float()                              # [Ttot, Hv=8]   fp32
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))     # [Ttot, Hv=8]   fp32   ∈ (0, 1]
    beta = torch.sigmoid(b.float())                              # [Ttot, Hv=8]   fp32   ∈ (0, 1)

    # GVA: broadcast 4 Q/K heads up to 8 V heads
    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)   # [Ttot, Hv=8, K=128]   bf16
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)   # [Ttot, Hv=8, K=128]   bf16

    output = torch.zeros(
        (total_seq_len, num_sab_heads, head_size), dtype=torch.bfloat16, device=device
    )                                                            # [Ttot, Hv=8, V=128]   bf16
    new_state = torch.zeros(
        (num_seqs, num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
    )                                                            # [N, Hv=8, V=128, K=128]   fp32

    # ────────────────────────────────────────────────────────────────────────
    # For each sequence: SERIAL scan over tokens, delta-rule update per token
    # ────────────────────────────────────────────────────────────────────────
    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())              # scalar
        seq_end = int(cu_seqlens[seq_idx + 1].item())            # scalar
        seq_len = seq_end - seq_start                            # scalar — this sequence's length

        if seq_len <= 0:
            continue

        # Load (or zero) the per-sequence state — stored K-last, operate K-first
        if state is not None:
            state_HKV = state[seq_idx].clone().float().transpose(-1, -2)  # [H,V,K] -> [Hv=8, K=128, V=128]  fp32
        else:
            state_HKV = torch.zeros(
                (num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
            )                                                    # [Hv=8, K=128, V=128]   fp32

        # SERIAL per-token recurrence (the bottleneck; chunk-parallel scan is the fix)
        for i in range(seq_len):
            t = seq_start + i                                    # global token index into Ttot axis

            q_H1K = q_exp[t].unsqueeze(1).float()                # [Hv=8, 1, K=128]    fp32
            k_H1K = k_exp[t].unsqueeze(1).float()                # [Hv=8, 1, K=128]    fp32
            v_H1V = v[t].unsqueeze(1).float()                    # [Hv=8, 1, V=128]    fp32
            g_H11 = g[t].unsqueeze(1).unsqueeze(2)               # [Hv=8, 1, 1]        fp32
            beta_H11 = beta[t].unsqueeze(1).unsqueeze(2)         # [Hv=8, 1, 1]        fp32

            # Decay: g * S_old
            old_state_HKV = g_H11 * state_HKV                    # [Hv=8, K=128, V=128]   fp32
            # Predict v from decayed state: old_v = k · g·S
            old_v_H1V = matmul(k_H1K, old_state_HKV)             # [H,1,K] @ [H,K,V] = [Hv=8, 1, V=128]  fp32
            # Mix with true v
            new_v_H1V = beta_H11 * v_H1V + (1 - beta_H11) * old_v_H1V  # [Hv=8, 1, V=128]   fp32
            # Erase kᵀ·old_v, insert kᵀ·new_v
            state_remove = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), old_v_H1V)
            #                    k_H1K.T: [Hv,K,1]  · old_v: [Hv,1,V]  →  [Hv=8, K=128, V=128]   fp32
            state_update = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), new_v_H1V)
            #                                                           →  [Hv=8, K=128, V=128]   fp32
            state_HKV = old_state_HKV - state_remove + state_update  # [Hv=8, K=128, V=128]      fp32

            # Output for this token: scale * q · S_new
            o_H1V = scale * matmul(q_H1K, state_HKV)             # [H,1,K] @ [H,K,V] = [Hv=8, 1, V=128]  fp32
            output[t] = o_H1V.squeeze(1).to(torch.bfloat16)      # [Hv=8, V=128]       bf16

        # Store final state in K-last layout
        new_state[seq_idx] = state_HKV.transpose(-1, -2)         # [H,K,V] -> [Hv=8, V=128, K=128]  fp32

    return output, new_state
