/*
 * GDN prefill — bf16 tensor-core chunked kernel (WY form).
 * V_TILE=16, C=32, 4 warps/block (warp-specialized matmuls).
 *
 * Within a chunk of C=32 tokens (starting state S_0, per-chunk γ_0 = 1):
 *   ŝ_t = S_t / γ_t  with γ_t = Π_{s≤t} g_s → un-gated delta-rule:
 *     ŝ_t = (I − β_t k_t k_t^T) ŝ_{t-1} + β_t k_t (v_t/γ_t)^T
 *   ⇒ ŝ_t = ŝ_0 + Σ_{s≤t} β_s k_s u_s^T
 *   where u is the triangular-solve residual of (I + L) u = ṽ − K ŝ_0,
 *     L[t,s] = β_s (k_t·k_s)  (strict lower), ṽ_t = v_t/γ_t.
 *   Output:  o_t = γ_t · (q_t^T S_0 + Σ_{s≤t} β_s (q_t·k_s) u_s)
 *   State:   S_C = γ_C · (S_0 + K^T · (β ⊙ u))
 *
 * Block: 4 warps (128 threads).  The four intra-chunk matmuls are split:
 *   warp 0: KS,  warp 1: QS,  warp 2: KK,  warp 3: QK.  Attn and KtU (state
 *   update) are split across warps by output tile.  cp.async.cg double-
 *   buffered Q/K/V loads overlap with compute.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

using namespace nvcuda;

constexpr int K_DIM  = 128;
constexpr int V_DIM  = 128;
constexpr int Hq     = 4;
constexpr int Hv     = 8;
constexpr int V_TILE = 16;
constexpr int N_V    = V_DIM / V_TILE;              // 8
constexpr int NT     = V_TILE / 16;                 // 1
constexpr int WARPS  = 4;
constexpr int BLOCK_THREADS = WARPS * 32;           // 128
constexpr int C      = 32;
constexpr int C_TILES = C / 16;                     // 2 (16×16 tiles along C dim)
// Pad shmem row strides to break the 128-stride bank conflict in ldmatrix.
// Choose strides ±8 elements (16 bytes) from natural sizes so that 16 rows
// of a ldmatrix tile hit 16 distinct 32-bit banks.
constexpr int K_PAD  = K_DIM + 16;  // 144 bf16 (288 B row, 32-B aligned)
constexpr int Vb_PAD = V_TILE + 16; // 32 bf16 for S_bf (64 B row)
// Row-stride pads for the small [C][V_TILE] / [C][C] scratch buffers that
// wmma::{load,store}_matrix_sync hits.  Natural strides (V_TILE=16 bf16,
// C=16 fp32) hit a power-of-two bank pattern that serializes the half-warp
// store into 2-way conflicts → 47–53 % excess wavefronts in ncu.  Push
// strides off the power-of-two boundary:
//   fp32 [16][16] → [16][20]   (80 B row, 20 banks wide)
//   bf16 [16][16] → [16][24]   (48 B row,  3 × 16 banks)
constexpr int VT_PAD_F32 = V_TILE + 4;   // 20 fp32
constexpr int VT_PAD_BF  = V_TILE + 8;   // 24 bf16
constexpr int C_PAD_F32  = C      + 4;   // 20 fp32
constexpr int C_PAD_BF   = C      + 8;   // 24 bf16

static __forceinline__ __device__ void cp_async_16(void* smem_dst, const void* gmem_src) {
  unsigned smem_int = __cvta_generic_to_shared(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :: "r"(smem_int), "l"(gmem_src));
}
static __forceinline__ __device__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}
static __forceinline__ __device__ void cp_async_wait_0() {
  asm volatile("cp.async.wait_group 0;\n");
}
static __forceinline__ __device__ void cp_async_wait_1() {
  asm volatile("cp.async.wait_group 1;\n");
}

__global__ __launch_bounds__(BLOCK_THREADS, 2)
void gdn_prefill_kernel(
    const __nv_bfloat16* __restrict__ q_ptr,
    const __nv_bfloat16* __restrict__ k_ptr,
    const __nv_bfloat16* __restrict__ v_ptr,
    const float*         __restrict__ state_ptr,
    const float*         __restrict__ A_log_ptr,
    const __nv_bfloat16* __restrict__ a_ptr,
    const float*         __restrict__ dt_bias_ptr,
    const __nv_bfloat16* __restrict__ b_ptr,
    const int64_t*       __restrict__ cu_seqlens,
    __nv_bfloat16*       __restrict__ out_ptr,
    float*               __restrict__ new_state,
    float scale)
{
  const int pid        = blockIdx.x;
  const int bh         = pid / N_V;
  const int v_tile_idx = pid % N_V;
  const int seq_idx    = bh / Hv;
  const int h_idx      = bh % Hv;
  const int qh         = (h_idx * Hq) / Hv;
  const int v_base     = v_tile_idx * V_TILE;
  const int tid        = threadIdx.x;
  const int warp_id    = tid >> 5;
  const int lane       = tid & 31;

  const int seq_start = (int)cu_seqlens[seq_idx];
  const int seq_end   = (int)cu_seqlens[seq_idx + 1];
  const int seq_len   = seq_end - seq_start;

  const float A_log   = A_log_ptr[h_idx];
  const float dt_bias = dt_bias_ptr[h_idx];

  // ----- Shared memory layout -----------------------------------------------
  // Static shmem stays under the 48 KB compile-time cap; the big Q/K/V
  // double-buffers live in dynamic shmem (launched with the right carveout).
  //
  // Static (~40 KB at C=32):
  __shared__ float         S_fp   [K_DIM][V_TILE];          //  8 KB
  __shared__ __nv_bfloat16 S_bf   [K_DIM][Vb_PAD];          //  8 KB
  __shared__ __nv_bfloat16 U_smem [C]    [VT_PAD_BF];       //  1.5 KB
  __shared__ __nv_bfloat16 Ubeta  [C]    [VT_PAD_BF];       //  1.5 KB
  __shared__ __nv_bfloat16 QKmb   [C]    [C_PAD_BF ];       //  2.5 KB
  __shared__ float         KS     [C]    [VT_PAD_F32];      //  2.5 KB
  __shared__ float         QS     [C]    [VT_PAD_F32];      //  2.5 KB
  __shared__ float         KK     [C]    [C_PAD_F32 ];      //  4.5 KB
  __shared__ float         QK     [C]    [C_PAD_F32 ];      //  4.5 KB
  __shared__ float         Attn   [C]    [VT_PAD_F32];      //  2.5 KB
  __shared__ float         gamma_cum[C];
  __shared__ float         g_arr    [C];
  __shared__ float         beta_arr [C];

  // Dynamic shmem: 2×C×K_PAD×2 B for Q + same for K + 2×C×V_TILE×2 B for V.
  extern __shared__ __nv_bfloat16 dyn_smem[];
  __nv_bfloat16 (*Q_smem)[C][K_PAD]  = reinterpret_cast<__nv_bfloat16 (*)[C][K_PAD]>(dyn_smem);
  __nv_bfloat16 (*K_smem)[C][K_PAD]  = reinterpret_cast<__nv_bfloat16 (*)[C][K_PAD]>(dyn_smem + 2 * C * K_PAD);
  __nv_bfloat16 (*V_smem)[C][V_TILE] = reinterpret_cast<__nv_bfloat16 (*)[C][V_TILE]>(dyn_smem + 4 * C * K_PAD);

  if (seq_len <= 0) {
    float* ns_bh = new_state + (seq_idx * Hv + h_idx) * V_DIM * K_DIM;
    if (state_ptr != nullptr) {
      const float* s_bh = state_ptr + (seq_idx * Hv + h_idx) * V_DIM * K_DIM;
      for (int i = tid; i < V_TILE * K_DIM; i += BLOCK_THREADS) {
        int vt = i / K_DIM, k = i % K_DIM;
        ns_bh[(v_base + vt) * K_DIM + k] = s_bh[(v_base + vt) * K_DIM + k];
      }
    } else {
      for (int i = tid; i < V_TILE * K_DIM; i += BLOCK_THREADS) {
        int vt = i / K_DIM, k = i % K_DIM;
        ns_bh[(v_base + vt) * K_DIM + k] = 0.f;
      }
    }
    return;
  }

  // ----- Load initial state (float4 per vt row) -----------------------------
  // K_DIM=128, 32 threads × float4 = 128 fp32 per row; use lane only (not warp_id)
  if (state_ptr != nullptr) {
    const float* s_bh = state_ptr + (seq_idx * Hv + h_idx) * V_DIM * K_DIM;
    // Distribute V_TILE rows across 4 warps (4 rows per warp)
    #pragma unroll
    for (int vt = warp_id; vt < V_TILE; vt += WARPS) {
      const float4* src4 =
          reinterpret_cast<const float4*>(s_bh + (v_base + vt) * K_DIM);
      float4 g = src4[lane];
      int k0 = lane * 4;
      S_fp[k0    ][vt] = g.x;
      S_fp[k0 + 1][vt] = g.y;
      S_fp[k0 + 2][vt] = g.z;
      S_fp[k0 + 3][vt] = g.w;
    }
  } else {
    #pragma unroll
    for (int vt = warp_id; vt < V_TILE; vt += WARPS) {
      int k0 = lane * 4;
      S_fp[k0    ][vt] = 0.f;
      S_fp[k0 + 1][vt] = 0.f;
      S_fp[k0 + 2][vt] = 0.f;
      S_fp[k0 + 3][vt] = 0.f;
    }
  }

  auto issue_chunk_load = [&](int buf, int chunk_start_l, int C_actual_l) {
    // Q, K: 256 float4 per matrix, all threads participate
    {
      constexpr int NVEC = (C * K_DIM) / 8;
      for (int i = tid; i < NVEC; i += BLOCK_THREADS) {
        int elem = i * 8;
        int tok  = elem / K_DIM;
        int kk   = elem % K_DIM;
        if (tok < C_actual_l) {
          int t = seq_start + chunk_start_l + tok;
          cp_async_16(&Q_smem[buf][tok][kk],
                      q_ptr + t * Hq * K_DIM + qh * K_DIM + kk);
          cp_async_16(&K_smem[buf][tok][kk],
                      k_ptr + t * Hq * K_DIM + qh * K_DIM + kk);
        } else {
          float4 zero = {0.f, 0.f, 0.f, 0.f};
          *reinterpret_cast<float4*>(&Q_smem[buf][tok][kk]) = zero;
          *reinterpret_cast<float4*>(&K_smem[buf][tok][kk]) = zero;
        }
      }
    }
    // V slab
    {
      constexpr int NVEC = (C * V_TILE) / 8;
      for (int i = tid; i < NVEC; i += BLOCK_THREADS) {
        int elem = i * 8;
        int tok  = elem / V_TILE;
        int vt   = elem % V_TILE;
        if (tok < C_actual_l) {
          int t = seq_start + chunk_start_l + tok;
          cp_async_16(&V_smem[buf][tok][vt],
                      v_ptr + t * Hv * V_DIM + h_idx * V_DIM + v_base + vt);
        } else {
          float4 zero = {0.f, 0.f, 0.f, 0.f};
          *reinterpret_cast<float4*>(&V_smem[buf][tok][vt]) = zero;
        }
      }
    }
  };

  // ----- Prologue: issue load for chunk 0 ------------------------------------
  int buf = 0;
  int C_actual0 = min(C, seq_len);
  issue_chunk_load(0, 0, C_actual0);
  cp_async_commit();

  // ===========================================================================
  // Chunk loop
  // ===========================================================================
  for (int chunk_start = 0; chunk_start < seq_len; chunk_start += C) {
    const int C_actual = min(C, seq_len - chunk_start);
    const int next     = chunk_start + C;
    const bool have_next = (next < seq_len);

    // Issue NEXT chunk's load (overlaps with compute below).
    if (have_next) {
      int C_next = min(C, seq_len - next);
      issue_chunk_load(1 - buf, next, C_next);
      cp_async_commit();
    }

    // ------------------------------------------------------------------------
    // OVERLAP WORK — runs in parallel with the in-flight cp.async for chunk N.
    // Uses S_fp (prev chunk's state) and global a,b (not Q/K/V).
    // ------------------------------------------------------------------------
    // Convert fp32 S → bf16 S — vectorized (float4 → 2 bf162)
    {
      constexpr int N_VEC = (K_DIM * V_TILE) / 4;
      for (int i = tid; i < N_VEC; i += BLOCK_THREADS) {
        int elem = i * 4;
        int k  = elem / V_TILE;
        int vt = elem % V_TILE;
        float4 f4 = *reinterpret_cast<float4*>(&S_fp[k][vt]);
        __nv_bfloat162 bf01 = __float22bfloat162_rn(make_float2(f4.x, f4.y));
        __nv_bfloat162 bf23 = __float22bfloat162_rn(make_float2(f4.z, f4.w));
        reinterpret_cast<__nv_bfloat162*>(&S_bf[k][vt])[0] = bf01;
        reinterpret_cast<__nv_bfloat162*>(&S_bf[k][vt])[1] = bf23;
      }
    }

    // Per-token scalars + warp-parallel γ cumprod
    if (warp_id == 0) {
      float g_local = 1.f, beta_local = 0.f;
      if (lane < C_actual) {
        int t = seq_start + chunk_start + lane;
        float a_val = __bfloat162float(a_ptr[t * Hv + h_idx]);
        float b_val = __bfloat162float(b_ptr[t * Hv + h_idx]);
        float x = a_val + dt_bias;
        float softplus = (x > 20.f) ? x : log1pf(expf(x));
        g_local    = expf(-expf(A_log) * softplus);
        beta_local = 1.f / (1.f + expf(-b_val));
      }
      const unsigned mask = 0xFFFFFFFFu;
      float cum = g_local;
      #pragma unroll
      for (int off = 1; off < C; off *= 2) {
        float up = __shfl_up_sync(mask, cum, off);
        if (lane >= off && lane < C) cum *= up;
      }
      if (lane < C) {
        gamma_cum[lane] = cum;
        beta_arr [lane] = beta_local;
        g_arr    [lane] = g_local;
      }
    }

    // Now wait for the CURRENT chunk's load (one outstanding if not last).
    if (have_next) {
      cp_async_wait_1();
    } else {
      cp_async_wait_0();
    }
    __syncthreads();

    // ṽ = v / γ  (no sync after — matmuls don't read V_smem)
    {
      constexpr int N_VEC = (C * V_TILE) / 2;
      for (int i = tid; i < N_VEC; i += BLOCK_THREADS) {
        int elem = i * 2;
        int tok  = elem / V_TILE;
        int vt   = elem % V_TILE;
        float inv_g = 1.f / gamma_cum[tok];
        __nv_bfloat162 v2 =
            *reinterpret_cast<__nv_bfloat162*>(&V_smem[buf][tok][vt]);
        float2 f2 = __bfloat1622float2(v2);
        f2.x *= inv_g; f2.y *= inv_g;
        *reinterpret_cast<__nv_bfloat162*>(&V_smem[buf][tok][vt]) =
            __float22bfloat162_rn(f2);
      }
    }

    // =======================================================================
    // Warp-specialized matmuls:
    //   warp 0: KS = K @ S     [C, K_DIM] @ [K_DIM, V_TILE]
    //   warp 1: QS = Q @ S
    //   warp 2: KK = K @ K^T   [C, K_DIM] @ [K_DIM, C]
    //   warp 3: QK = Q @ K^T
    // =======================================================================
    if (warp_id == 0) {
      // KS = K @ S  [C, K_DIM] @ [K_DIM, V_TILE]  → [C, V_TILE] = C_TILES row-tiles.
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> aK[C_TILES];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bS;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[C_TILES];
      #pragma unroll
      for (int i = 0; i < C_TILES; ++i) wmma::fill_fragment(c[i], 0.f);
      #pragma unroll
      for (int k_off = 0; k_off < K_DIM; k_off += 16) {
        wmma::load_matrix_sync(bS, &S_bf[k_off][0], Vb_PAD);
        #pragma unroll
        for (int i = 0; i < C_TILES; ++i) {
          wmma::load_matrix_sync(aK[i], &K_smem[buf][i * 16][k_off], K_PAD);
          wmma::mma_sync(c[i], aK[i], bS, c[i]);
        }
      }
      #pragma unroll
      for (int i = 0; i < C_TILES; ++i)
        wmma::store_matrix_sync(&KS[i * 16][0], c[i], VT_PAD_F32, wmma::mem_row_major);
    } else if (warp_id == 1) {
      // QS = Q @ S  — same shape as KS.
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> aQ[C_TILES];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bS;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[C_TILES];
      #pragma unroll
      for (int i = 0; i < C_TILES; ++i) wmma::fill_fragment(c[i], 0.f);
      #pragma unroll
      for (int k_off = 0; k_off < K_DIM; k_off += 16) {
        wmma::load_matrix_sync(bS, &S_bf[k_off][0], Vb_PAD);
        #pragma unroll
        for (int i = 0; i < C_TILES; ++i) {
          wmma::load_matrix_sync(aQ[i], &Q_smem[buf][i * 16][k_off], K_PAD);
          wmma::mma_sync(c[i], aQ[i], bS, c[i]);
        }
      }
      #pragma unroll
      for (int i = 0; i < C_TILES; ++i)
        wmma::store_matrix_sync(&QS[i * 16][0], c[i], VT_PAD_F32, wmma::mem_row_major);
    } else if (warp_id == 2) {
      // KK = K @ K^T  [C, K_DIM] @ [K_DIM, C] → [C, C] = C_TILES × C_TILES tiles.
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> aK[C_TILES];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bK[C_TILES];
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[C_TILES][C_TILES];
      #pragma unroll
      for (int i = 0; i < C_TILES; ++i)
        #pragma unroll
        for (int j = 0; j < C_TILES; ++j) wmma::fill_fragment(c[i][j], 0.f);
      #pragma unroll
      for (int k_off = 0; k_off < K_DIM; k_off += 16) {
        #pragma unroll
        for (int i = 0; i < C_TILES; ++i) {
          wmma::load_matrix_sync(aK[i], &K_smem[buf][i * 16][k_off], K_PAD);
          wmma::load_matrix_sync(bK[i], &K_smem[buf][i * 16][k_off], K_PAD);
        }
        #pragma unroll
        for (int i = 0; i < C_TILES; ++i)
          #pragma unroll
          for (int j = 0; j < C_TILES; ++j)
            wmma::mma_sync(c[i][j], aK[i], bK[j], c[i][j]);
      }
      #pragma unroll
      for (int i = 0; i < C_TILES; ++i)
        #pragma unroll
        for (int j = 0; j < C_TILES; ++j)
          wmma::store_matrix_sync(&KK[i * 16][j * 16], c[i][j], C_PAD_F32, wmma::mem_row_major);
    } else /* warp_id == 3 */ {
      // QK = Q @ K^T — same shape as KK.
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> aQ[C_TILES];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bK[C_TILES];
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[C_TILES][C_TILES];
      #pragma unroll
      for (int i = 0; i < C_TILES; ++i)
        #pragma unroll
        for (int j = 0; j < C_TILES; ++j) wmma::fill_fragment(c[i][j], 0.f);
      #pragma unroll
      for (int k_off = 0; k_off < K_DIM; k_off += 16) {
        #pragma unroll
        for (int i = 0; i < C_TILES; ++i) {
          wmma::load_matrix_sync(aQ[i], &Q_smem[buf][i * 16][k_off], K_PAD);
          wmma::load_matrix_sync(bK[i], &K_smem[buf][i * 16][k_off], K_PAD);
        }
        #pragma unroll
        for (int i = 0; i < C_TILES; ++i)
          #pragma unroll
          for (int j = 0; j < C_TILES; ++j)
            wmma::mma_sync(c[i][j], aQ[i], bK[j], c[i][j]);
      }
      #pragma unroll
      for (int i = 0; i < C_TILES; ++i)
        #pragma unroll
        for (int j = 0; j < C_TILES; ++j)
          wmma::store_matrix_sync(&QK[i * 16][j * 16], c[i][j], C_PAD_F32, wmma::mem_row_major);
    }
    __syncthreads();

    // =======================================================================
    // Triangular solve for u — warp 0 only (serial across t, lanes = vt).
    // Fused: also writes U_smem and Ubeta (β⊙u).
    // =======================================================================
    if (warp_id == 0) {
      float u_reg[C];
      #pragma unroll
      for (int t = 0; t < C; ++t) u_reg[t] = 0.f;
      if (lane < V_TILE) {
        const int vt = lane;
        #pragma unroll
        for (int t = 0; t < C; ++t) {
          float v_val = __bfloat162float(V_smem[buf][t][vt]);
          float u_val = v_val - KS[t][vt];
          #pragma unroll
          for (int s = 0; s < C; ++s) {
            if (s < t) u_val -= beta_arr[s] * KK[t][s] * u_reg[s];
          }
          u_reg[t]      = u_val;
          U_smem[t][vt] = __float2bfloat16(u_val);
          Ubeta [t][vt] = __float2bfloat16(beta_arr[t] * u_val);
        }
      }
    }
    // QKmb = causal(QK) ⊙ β  (other warps do this while warp 0 solves)
    if (warp_id != 0) {
      for (int i = tid - 32; i < C * C; i += BLOCK_THREADS - 32) {
        if (i >= 0) {
          int t = i / C, s = i % C;
          float val = (s <= t) ? (QK[t][s] * beta_arr[s]) : 0.f;
          QKmb[t][s] = __float2bfloat16(val);
        }
      }
    }
    __syncthreads();

    // =======================================================================
    // Matmul 5: Attn = QKmb @ U_smem  [C, C] @ [C, V_TILE]  — warp 0 only
    // =======================================================================
    // Matmul 5: Attn = QKmb @ U_smem — warp 0 only.
    // Shape: [C, C] @ [C, V_TILE] = [C, V_TILE].
    // With C=32, V_TILE=16, this is C_TILES row-tiles × (C_TILES inner reductions).
    if (warp_id == 0) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[C_TILES];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[C_TILES];
      #pragma unroll
      for (int i = 0; i < C_TILES; ++i) wmma::fill_fragment(c[i], 0.f);
      #pragma unroll
      for (int k_off = 0; k_off < C; k_off += 16) {
        wmma::load_matrix_sync(b, &U_smem[k_off][0], VT_PAD_BF);
        #pragma unroll
        for (int i = 0; i < C_TILES; ++i) {
          wmma::load_matrix_sync(a[i], &QKmb[i * 16][k_off], C_PAD_BF);
          wmma::mma_sync(c[i], a[i], b, c[i]);
        }
      }
      #pragma unroll
      for (int i = 0; i < C_TILES; ++i)
        wmma::store_matrix_sync(&Attn[i * 16][0], c[i], VT_PAD_F32, wmma::mem_row_major);
    }
    __syncthreads();

    // Output write — all threads (256 elements / 128 threads = 2 per thread).
    {
      constexpr int N_VEC = (C * V_TILE) / 2;
      for (int i = tid; i < N_VEC; i += BLOCK_THREADS) {
        int elem = i * 2;
        int t    = elem / V_TILE;
        int vt   = elem % V_TILE;
        if (t < C_actual) {
          int token = seq_start + chunk_start + t;
          float s_g = scale * gamma_cum[t];
          float o0 = s_g * (QS[t][vt]   + Attn[t][vt]);
          float o1 = s_g * (QS[t][vt+1] + Attn[t][vt+1]);
          __nv_bfloat162 o2 = __float22bfloat162_rn(make_float2(o0, o1));
          *reinterpret_cast<__nv_bfloat162*>(
              out_ptr + token * Hv * V_DIM + h_idx * V_DIM + v_base + vt) = o2;
        }
      }
    }

    // State update — 8 m-tiles (K_DIM/16), 2 per warp.
    // Each tile reduces over chunk dim C via C_TILES=2 WMMA ops.
    // S += K^T[:, m_off:m_off+16] @ Ubeta[:, 0:V_TILE], scaled by γ_C.
    {
      const float gC = gamma_cum[C - 1];
      const int m_base = warp_id * 32;
      #pragma unroll
      for (int dm = 0; dm < 32; dm += 16) {
        int m_off = m_base + dm;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::load_matrix_sync(c, &S_fp[m_off][0], V_TILE, wmma::mem_row_major);
        #pragma unroll
        for (int k_chunk = 0; k_chunk < C; k_chunk += 16) {
          wmma::load_matrix_sync(a, &K_smem[buf][k_chunk][m_off], K_PAD);
          wmma::load_matrix_sync(b, &Ubeta[k_chunk][0], VT_PAD_BF);
          wmma::mma_sync(c, a, b, c);
        }
        #pragma unroll
        for (int i = 0; i < c.num_elements; ++i) c.x[i] *= gC;
        wmma::store_matrix_sync(&S_fp[m_off][0], c, V_TILE, wmma::mem_row_major);
      }
    }
    __syncthreads();

    buf = 1 - buf;
  }

  // ===========================================================================
  // Final state write-back — vectorized float4 stores (V_TILE rows across warps)
  // ===========================================================================
  float* ns_bh = new_state + (seq_idx * Hv + h_idx) * V_DIM * K_DIM;
  #pragma unroll
  for (int vt = warp_id; vt < V_TILE; vt += WARPS) {
    int k0 = lane * 4;
    float4 g;
    g.x = S_fp[k0    ][vt];
    g.y = S_fp[k0 + 1][vt];
    g.z = S_fp[k0 + 2][vt];
    g.w = S_fp[k0 + 3][vt];
    float4* dst4 = reinterpret_cast<float4*>(ns_bh + (v_base + vt) * K_DIM);
    dst4[lane] = g;
  }
}

// ---------------------------------------------------------------------------
// tvm-ffi entry point
// ---------------------------------------------------------------------------

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

static void run_impl(
    TensorView q, TensorView k, TensorView v,
    Optional<TensorView> state,
    TensorView A_log, TensorView a, TensorView dt_bias, TensorView b,
    TensorView cu_seqlens,
    double scale,
    TensorView output, TensorView new_state)
{
  const int64_t N = cu_seqlens.shape()[0] - 1;
  float scale_f = (scale == 0.0) ? (1.0f / sqrtf((float)K_DIM)) : (float)scale;

  const float* state_data =
      state.has_value()
          ? static_cast<const float*>(state.value().data_ptr())
          : nullptr;

  dim3 grid(N * Hv * N_V);
  dim3 block(BLOCK_THREADS);

  // Dynamic shmem holds Q/K/V double buffers for C=32.
  constexpr int K_PAD_H = K_DIM + 16;
  constexpr int DYN_BYTES =
      (2 * 32 * K_PAD_H * 2) +   // Q:  2 × C × K_PAD × sizeof(bf16)
      (2 * 32 * K_PAD_H * 2) +   // K
      (2 * 32 * 16 * 2);         // V:  2 × C × V_TILE × sizeof(bf16)

  static bool set_shmem_attr = false;
  if (!set_shmem_attr) {
    cudaFuncSetAttribute(gdn_prefill_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         DYN_BYTES);
    set_shmem_attr = true;
  }

  gdn_prefill_kernel<<<grid, block, DYN_BYTES, 0>>>(
      static_cast<const __nv_bfloat16*>(q.data_ptr()),
      static_cast<const __nv_bfloat16*>(k.data_ptr()),
      static_cast<const __nv_bfloat16*>(v.data_ptr()),
      state_data,
      static_cast<const float*>(A_log.data_ptr()),
      static_cast<const __nv_bfloat16*>(a.data_ptr()),
      static_cast<const float*>(dt_bias.data_ptr()),
      static_cast<const __nv_bfloat16*>(b.data_ptr()),
      static_cast<const int64_t*>(cu_seqlens.data_ptr()),
      static_cast<__nv_bfloat16*>(output.data_ptr()),
      static_cast<float*>(new_state.data_ptr()),
      scale_f);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_prefill, run_impl);
