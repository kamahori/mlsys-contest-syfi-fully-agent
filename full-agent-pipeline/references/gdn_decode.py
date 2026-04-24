# Benchmark baseline — reference implementation embedded in:
#   mlsys26-contest/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json
# Definition: gdn_decode_qk4_v8_d128_k_last
#
# This is the EXACT Python code the flashinfer-bench harness times
# our solution against. Reported "speedup" = ref_latency / our_latency.
# Kept simple to define correctness, not optimized.
#
# Extracted verbatim (2026-04-19) so it can be diffed against the
# optimized solution/python/<name> without spelunking the JSON.
#
# SHAPE KEY (all from definitions/.../json; const values inlined):
#   B  = batch_size  (var)
#   T  = seq_len = 1                 (const, always single-token decode)
#   Hq = num_q_heads = 4             (query  / key heads; GVA)
#   Hk = num_k_heads = 4
#   Hv = num_v_heads = 8             (value heads — state is per V head)
#   K  = V = head_size = 128         (same for Q, K, V)
#   GVA repeat factor: Hv/Hq = Hv/Hk = 2   (each Q/K head shared across 2 V heads)

import math
import torch
import torch.nn.functional as F


def matmul(a: torch.Tensor, b: torch.Tensor):
    """Float32 matmul for numerical stability."""
    return a.float() @ b.float()


@torch.no_grad()
def run(
    q,          # [B, T=1, Hq=4, K=128]          bf16
    k,          # [B, T=1, Hk=4, K=128]          bf16
    v,          # [B, T=1, Hv=8, V=128]          bf16
    state,      # [B, Hv=8, V=128, K=128]        fp32   (K-LAST layout on disk)
    A_log,      # [Hv=8]                         fp32   (learned per-head decay)
    a,          # [B, T=1, Hv=8]                 bf16   (per-token gate logit)
    dt_bias,    # [Hv=8]                         fp32   (learned per-head bias)
    b,          # [B, T=1, Hv=8]                 bf16   (per-token mixing logit)
    scale,      # scalar fp32 (if None/0 → 1/√K)
):              # returns (output [B, T=1, Hv=8, V=128] bf16, new_state [B, Hv=8, V=128, K=128] fp32)
    """
    Gated Delta Net decode reference implementation (k-last layout).

    State layout: [B, H, V, K] (k-last, K dimension at the end)

    Gate computation:
    g = exp(-exp(A_log) * softplus(a + dt_bias))
    beta = sigmoid(b)

    Delta rule update:
    state_new = g * state_old + k^T @ (beta * v + (1-beta) * k @ state_old) - k^T @ (k @ state_old)
    output = scale * q @ state_new
    """
    B, T, num_q_heads, K = q.shape                               # T=1, num_q_heads=4, K=128
    _, _, num_k_heads, _ = k.shape                               # num_k_heads=4
    _, _, num_v_heads, V = v.shape                               # num_v_heads=8, V=128
    num_heads = num_v_heads                                      # 8
    device = q.device

    assert num_q_heads == 4
    assert num_k_heads == 4
    assert num_v_heads == 8
    assert K == 128 and V == 128
    assert T == 1

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)                               # scalar fp32

    # ────────────────────────────────────────────────────────────────────────
    # Gates (g: decay, beta: mixing rate)
    # ────────────────────────────────────────────────────────────────────────
    x = a.float() + dt_bias.float()                              # [B, 1, Hv=8]    fp32  (dt_bias broadcast)
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))     # [B, 1, Hv=8]    fp32   ∈ (0, 1]
    beta = torch.sigmoid(b.float())                              # [B, 1, Hv=8]    fp32   ∈ (0, 1)

    # Squeeze the T=1 axis and cast; also float-ify state
    q_f32 = q.squeeze(1).float()                                 # [B, Hq=4, K=128]   fp32
    k_f32 = k.squeeze(1).float()                                 # [B, Hk=4, K=128]   fp32
    v_f32 = v.squeeze(1).float()                                 # [B, Hv=8, V=128]   fp32
    g_f32 = g.squeeze(1).float()                                 # [B, Hv=8]          fp32
    beta_f32 = beta.squeeze(1).float()                           # [B, Hv=8]          fp32

    if state is not None:
        state_f32 = state.float()                                # [B, Hv=8, V=128, K=128]  fp32  (K-last)
    else:
        state_f32 = torch.zeros(B, num_heads, V, K, dtype=torch.float32, device=device)

    # GVA: broadcast 4 Q/K heads up to 8 V heads by repeat_interleave(2, dim=1)
    q_exp = q_f32.repeat_interleave(num_v_heads // num_q_heads, dim=1)   # [B, Hv=8, K=128]   fp32
    k_exp = k_f32.repeat_interleave(num_v_heads // num_k_heads, dim=1)   # [B, Hv=8, K=128]   fp32

    new_state = torch.zeros_like(state_f32)                      # [B, Hv=8, V=128, K=128]  fp32
    output = torch.zeros(B, num_heads, V, dtype=torch.float32, device=device)  # [B, Hv=8, V=128]  fp32

    # ────────────────────────────────────────────────────────────────────────
    # Delta-rule update, per (batch, head)
    # ────────────────────────────────────────────────────────────────────────
    for b_idx in range(B):
        for h_idx in range(num_heads):                           # 0..Hv-1
            q_h = q_exp[b_idx, h_idx]                            # [K=128]               fp32
            k_h = k_exp[b_idx, h_idx]                            # [K=128]               fp32
            v_h = v_f32[b_idx, h_idx]                            # [V=128]               fp32
            h_state = state_f32[b_idx, h_idx].clone().transpose(-1, -2)  # [V,K] -> [K=128, V=128]  fp32
            g_val = g_f32[b_idx, h_idx]                          # scalar fp32
            beta_val = beta_f32[b_idx, h_idx]                    # scalar fp32

            # Decay:           g * S_old
            old_state = g_val * h_state                          # [K=128, V=128]        fp32

            # Predict v from decayed state using current k
            old_v = k_h @ old_state                              # [K=128] @ [K=128, V=128] = [V=128]   fp32

            # Mix prediction with true v (β=1 → fully trust true v)
            new_v = beta_val * v_h + (1 - beta_val) * old_v      # [V=128]               fp32

            # Erase the old kᵀ·old_v write, insert kᵀ·new_v
            state_remove = k_h.unsqueeze(1) @ old_v.unsqueeze(0)  # [K,1] @ [1,V] = [K=128, V=128]  fp32
            state_update = k_h.unsqueeze(1) @ new_v.unsqueeze(0)  # [K,1] @ [1,V] = [K=128, V=128]  fp32
            h_state = old_state - state_remove + state_update    # [K=128, V=128]        fp32
            # Equivalent closed form:  S_new = g*S_old + β * kᵀ · (v − k·(g*S_old))

            # Output this head: scale * q · S_new
            output[b_idx, h_idx] = scale * (q_h @ h_state)       # [K=128] @ [K,V] = [V=128]  fp32

            # Store back in K-last layout
            new_state[b_idx, h_idx] = h_state.transpose(-1, -2)  # [K,V] -> [V=128, K=128]  fp32

    output = output.unsqueeze(1).to(torch.bfloat16)              # [B, T=1, Hv=8, V=128]  bf16
    return output, new_state                                     # new_state: [B, Hv=8, V=128, K=128] fp32
