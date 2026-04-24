/*
 * GDN prefill — bf16 tensor-core chunked kernel (WY form).
 *
 * Within a chunk of C=16 tokens (starting state S_0, per-chunk gamma_0 = 1):
 *   s_hat_t = S_t / gamma_t  with gamma_t = Π_{s<=t} g_s → un-gated delta-rule:
 *     s_hat_t = (I - beta_t k_t k_t^T) s_hat_{t-1} + beta_t k_t (v_t/gamma_t)^T
 *   ⇒ s_hat_t = s_hat_0 + Σ_{s<=t} beta_s k_s u_s^T
 *   where u is the triangular-solve residual of (I + L) u = v_tilde - K s_hat_0,
 *     L[t,s] = beta_s (k_t·k_s)  (strict lower), v_tilde_t = v_t/gamma_t.
 *   Output: o_t = gamma_t · (q_t^T S_0 + Σ_{s<=t} beta_s (q_t·k_s) u_s)
 *   State:  S_C = gamma_C · (S_0 + K^T · (beta ⊙ u))
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
constexpr int WARPS  = 4;
constexpr int BLOCK_THREADS = WARPS * 32;           // 128
constexpr int C      = 16;
// Pad shmem row strides to break the 128-stride bank conflict in ldmatrix.
constexpr int K_PAD  = K_DIM + 16;                  // 144 bf16
constexpr int Vb_PAD = V_TILE + 16;                 // 32 bf16

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

__global__ __launch_bounds__(BLOCK_THREADS, 4)
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

  // Shared memory (~36KB)
  __shared__ float         S_fp   [K_DIM][V_TILE];
  __shared__ __nv_bfloat16 S_bf   [K_DIM][Vb_PAD];
  __shared__ __nv_bfloat16 Q_smem [2][C][K_PAD];
  __shared__ __nv_bfloat16 K_smem [2][C][K_PAD];
  __shared__ __nv_bfloat16 V_smem [2][C][V_TILE];
  __shared__ __nv_bfloat16 U_smem [C    ][V_TILE];
  __shared__ __nv_bfloat16 Ubeta  [C    ][V_TILE];
  __shared__ __nv_bfloat16 QKmb   [C    ][C    ];
  __shared__ float         KS     [C    ][V_TILE];
  __shared__ float         QS     [C    ][V_TILE];
  __shared__ float         KK     [C    ][C    ];
  __shared__ float         Attn   [C    ][V_TILE];
  __shared__ float         gamma_cum[C];
  __shared__ float         g_arr    [C];
  __shared__ float         beta_arr [C];

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

  // Load initial state (float4 per vt row)
  if (state_ptr != nullptr) {
    const float* s_bh = state_ptr + (seq_idx * Hv + h_idx) * V_DIM * K_DIM;
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

  // Persistent state fragments (held in registers across chunks)
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_state[2];
  __syncthreads();
  #pragma unroll
  for (int t_idx = 0; t_idx < 2; ++t_idx) {
    int m_off = warp_id * 32 + t_idx * 16;
    wmma::load_matrix_sync(c_state[t_idx], &S_fp[m_off][0], V_TILE, wmma::mem_row_major);
  }
  // Initial S_fp → S_bf conversion
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

  // Prologue: issue load for chunk 0
  int buf = 0;
  int C_actual0 = min(C, seq_len);
  issue_chunk_load(0, 0, C_actual0);
  cp_async_commit();

  // Chunk loop
  for (int chunk_start = 0; chunk_start < seq_len; chunk_start += C) {
    const int C_actual = min(C, seq_len - chunk_start);
    const int next     = chunk_start + C;
    const bool have_next = (next < seq_len);

    if (have_next) {
      int C_next = min(C, seq_len - next);
      issue_chunk_load(1 - buf, next, C_next);
      cp_async_commit();
    }

    // Per-token scalars + warp-parallel gamma cumprod
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

    // Wait for current chunk's load
    if (have_next) {
      cp_async_wait_1();
    } else {
      cp_async_wait_0();
    }
    __syncthreads();

    // v_tilde = v / gamma
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

    // Warp-specialized matmuls:
    //   warp 0: KS = K @ S     [C, K_DIM] @ [K_DIM, V_TILE]
    //   warp 1: QS = Q @ S
    //   warp 2: KK = K @ K^T   [C, K_DIM] @ [K_DIM, C]
    //   warp 3: QK = Q @ K^T
    if (warp_id == 0) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> aK;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bS;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      wmma::fill_fragment(c, 0.f);
      #pragma unroll
      for (int k_off = 0; k_off < K_DIM; k_off += 16) {
        wmma::load_matrix_sync(aK, &K_smem[buf][0][k_off], K_PAD);
        wmma::load_matrix_sync(bS, &S_bf[k_off][0], Vb_PAD);
        wmma::mma_sync(c, aK, bS, c);
      }
      wmma::store_matrix_sync(&KS[0][0], c, V_TILE, wmma::mem_row_major);
    } else if (warp_id == 1) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> aQ;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bS;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      wmma::fill_fragment(c, 0.f);
      #pragma unroll
      for (int k_off = 0; k_off < K_DIM; k_off += 16) {
        wmma::load_matrix_sync(aQ, &Q_smem[buf][0][k_off], K_PAD);
        wmma::load_matrix_sync(bS, &S_bf[k_off][0], Vb_PAD);
        wmma::mma_sync(c, aQ, bS, c);
      }
      wmma::store_matrix_sync(&QS[0][0], c, V_TILE, wmma::mem_row_major);
    } else if (warp_id == 2) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> aK;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bK;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      wmma::fill_fragment(c, 0.f);
      #pragma unroll
      for (int k_off = 0; k_off < K_DIM; k_off += 16) {
        wmma::load_matrix_sync(aK, &K_smem[buf][0][k_off], K_PAD);
        wmma::load_matrix_sync(bK, &K_smem[buf][0][k_off], K_PAD);
        wmma::mma_sync(c, aK, bK, c);
      }
      wmma::store_matrix_sync(&KK[0][0], c, C, wmma::mem_row_major);
    } else /* warp_id == 3 */ {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> aQ;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bK;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      wmma::fill_fragment(c, 0.f);
      #pragma unroll
      for (int k_off = 0; k_off < K_DIM; k_off += 16) {
        wmma::load_matrix_sync(aQ, &Q_smem[buf][0][k_off], K_PAD);
        wmma::load_matrix_sync(bK, &K_smem[buf][0][k_off], K_PAD);
        wmma::mma_sync(c, aQ, bK, c);
      }
      // Direct fragment → QKmb write with causal mask + beta multiply.
      const int group = lane >> 2;
      const int tip   = lane & 3;
      const int s0 = tip * 2;
      const int s8 = tip * 2 + 8;
      auto pack = [&](int t, int s, float f0, float f1) -> __nv_bfloat162 {
        float v0 = (s     <= t) ? (f0 * beta_arr[s])     : 0.f;
        float v1 = (s + 1 <= t) ? (f1 * beta_arr[s + 1]) : 0.f;
        return __float22bfloat162_rn(make_float2(v0, v1));
      };
      *reinterpret_cast<__nv_bfloat162*>(&QKmb[group    ][s0]) = pack(group    , s0, c.x[0], c.x[1]);
      *reinterpret_cast<__nv_bfloat162*>(&QKmb[group + 8][s0]) = pack(group + 8, s0, c.x[2], c.x[3]);
      *reinterpret_cast<__nv_bfloat162*>(&QKmb[group    ][s8]) = pack(group    , s8, c.x[4], c.x[5]);
      *reinterpret_cast<__nv_bfloat162*>(&QKmb[group + 8][s8]) = pack(group + 8, s8, c.x[6], c.x[7]);
    }
    __syncthreads();

    // Triangular solve for u — warp 0 only
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
    __syncthreads();

    // Matmul 5: Attn = QKmb @ U_smem  [C, C] @ [C, V_TILE]  — warp 0 only
    if (warp_id == 0) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      wmma::fill_fragment(c, 0.f);
      wmma::load_matrix_sync(a, &QKmb[0][0], C);
      wmma::load_matrix_sync(b, &U_smem[0][0], V_TILE);
      wmma::mma_sync(c, a, b, c);
      wmma::store_matrix_sync(&Attn[0][0], c, V_TILE, wmma::mem_row_major);
    }
    __syncthreads();

    // Output write — all threads
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

    // State update — 8 m-tiles, 2 per warp. c_state persistent across chunks.
    {
      const float gC = gamma_cum[C - 1];
      const int m_base = warp_id * 32;
      const int group = lane >> 2;
      const int tip   = lane & 3;
      const int v0 = tip * 2;
      const int v8 = tip * 2 + 8;
      #pragma unroll
      for (int t_idx = 0; t_idx < 2; ++t_idx) {
        int m_off = m_base + t_idx * 16;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
        wmma::load_matrix_sync(a, &K_smem[buf][0][m_off], K_PAD);
        wmma::load_matrix_sync(b, &Ubeta[0][0], V_TILE);
        wmma::mma_sync(c_state[t_idx], a, b, c_state[t_idx]);
        #pragma unroll
        for (int i = 0; i < c_state[t_idx].num_elements; ++i) c_state[t_idx].x[i] *= gC;
        int k_r0 = m_off + group;
        int k_r1 = m_off + group + 8;
        __nv_bfloat162 p01 = __float22bfloat162_rn(make_float2(c_state[t_idx].x[0], c_state[t_idx].x[1]));
        __nv_bfloat162 p23 = __float22bfloat162_rn(make_float2(c_state[t_idx].x[2], c_state[t_idx].x[3]));
        __nv_bfloat162 p45 = __float22bfloat162_rn(make_float2(c_state[t_idx].x[4], c_state[t_idx].x[5]));
        __nv_bfloat162 p67 = __float22bfloat162_rn(make_float2(c_state[t_idx].x[6], c_state[t_idx].x[7]));
        *reinterpret_cast<__nv_bfloat162*>(&S_bf[k_r0][v0]) = p01;
        *reinterpret_cast<__nv_bfloat162*>(&S_bf[k_r1][v0]) = p23;
        *reinterpret_cast<__nv_bfloat162*>(&S_bf[k_r0][v8]) = p45;
        *reinterpret_cast<__nv_bfloat162*>(&S_bf[k_r1][v8]) = p67;
      }
    }
    __syncthreads();

    buf = 1 - buf;
  }

  // Final state write-back
  #pragma unroll
  for (int t_idx = 0; t_idx < 2; ++t_idx) {
    int m_off = warp_id * 32 + t_idx * 16;
    wmma::store_matrix_sync(&S_fp[m_off][0], c_state[t_idx], V_TILE, wmma::mem_row_major);
  }
  __syncthreads();
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
    TensorView q,
    TensorView k,
    TensorView v,
    Optional<TensorView> state,
    TensorView A_log,
    TensorView a,
    TensorView dt_bias,
    TensorView b,
    TensorView cu_seqlens,
    double scale,
    TensorView output,
    TensorView new_state)
{
  const int64_t N = cu_seqlens.shape()[0] - 1;
  if (N <= 0) return;

  float scale_f = (scale == 0.0) ? (1.0f / sqrtf((float)K_DIM)) : (float)scale;

  const float* state_data =
      state.has_value()
          ? static_cast<const float*>(state.value().data_ptr())
          : nullptr;

  dim3 grid((unsigned)(N * Hv * N_V));
  dim3 block(BLOCK_THREADS);

  gdn_prefill_kernel<<<grid, block, 0, 0>>>(
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
