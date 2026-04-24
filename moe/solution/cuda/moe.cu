/*
 * Fused FP8 block-scale MoE — full CUDA implementation from scratch.
 *
 * Pipeline:
 *   1. route_kernel           — DeepSeek-V3 no-aux routing
 *   2. perm_count_kernel      — counts[le] per local expert
 *   3. scan_offsets_kernel    — offs[E_LOCAL+1] and write_ptr
 *   4. perm_fill_kernel       — perm_token[N_sel], perm_weight[N_sel]
 *   5. dequant_hs_kernel      — hidden_states fp8 -> bf16 [T, H]
 *   6. gather_hs_kernel       — A_perm[N_sel, H] = hs_bf16[perm_token]
 *   7. dequant_w_kernel       — W13, W2 fp8 -> bf16
 *   8. gemm_bf16_kernel       — A @ B^T, bf16 tensor cores (WMMA)
 *   9. swiglu_kernel          — silu(X2)*X1 -> C[Tk, I]
 *  10. fin_acc_kernel         — fp32 atomic-add weighted scatter
 *  11. fin_cast_kernel        — fp32 -> bf16 output
 *
 * Constants (DeepSeek-V3/R1, moe_fp8_block_scale track):
 *   H=7168, I=2048, BS=128, E_LOCAL=32, E_GLOBAL=256,
 *   TOP_K=8, N_GROUP=8, TOPK_GROUP=4
 */

#include <cuda_bf16.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <mma.h>
#include <math.h>
#include <math_constants.h>
#include <stdint.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

using namespace nvcuda;

namespace {

constexpr int H          = 7168;
constexpr int I          = 2048;
constexpr int II         = 2 * I;               // 4096
constexpr int E_LOCAL    = 32;
constexpr int E_GLOBAL   = 256;
constexpr int TOP_K      = 8;
constexpr int N_GROUP    = 8;
constexpr int TOPK_GROUP = 4;
constexpr int GROUP_SIZE = E_GLOBAL / N_GROUP;  // 32
constexpr int BS         = 128;                 // quant block size
constexpr int NH_BLKS    = H / BS;              // 56
constexpr int NI_BLKS    = I / BS;              // 16
constexpr int NII_BLKS   = II / BS;             // 32

#define NEG_INF_F (-CUDART_INF_F)

// ======================================================================
// Kernel 1: routing.
// One CTA per token, 256 threads. One thread per expert.
// Uses argmax-with-lane-tiebreak so top-2 group scores are correct with ties.
// ======================================================================
__global__ void route_kernel(
    const float* __restrict__ logits,          // [T, E_GLOBAL]
    const __nv_bfloat16* __restrict__ bias,    // [E_GLOBAL]
    float scaling_factor,
    int T,
    int local_expert_offset,
    int32_t* __restrict__ topk_idx_out,        // [T, TOP_K]
    float*   __restrict__ topk_w_out,          // [T, TOP_K]
    int32_t* __restrict__ counts)              // [E_LOCAL]
{
  const int t = blockIdx.x;
  if (t >= T) return;
  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  __shared__ float s_sig[E_GLOBAL];
  __shared__ float s_wb [E_GLOBAL];
  __shared__ float gs[N_GROUP];
  __shared__ int   group_sel[N_GROUP];
  __shared__ int   tk_idx[TOP_K];
  __shared__ float tk_sig[TOP_K];
  __shared__ int   s_best_idx;
  __shared__ float warp_v[8];
  __shared__ int   warp_i[8];

  // sigmoid + add bias
  float lg = logits[(size_t)t * E_GLOBAL + tid];
  float si = 1.0f / (1.0f + __expf(-lg));
  s_sig[tid] = si;
  float b = __bfloat162float(bias[tid]);
  s_wb[tid] = si + b;
  if (tid < N_GROUP) group_sel[tid] = 0;
  __syncthreads();

  // per-warp top-2 sum: argmax tiebreak by (value, -lane)
  {
    float v  = s_wb[warp * GROUP_SIZE + lane];
    float bv = v;
    int   bl = lane;
    #pragma unroll
    for (int m = 16; m >= 1; m >>= 1) {
      float ov = __shfl_xor_sync(0xffffffff, bv, m);
      int   ol = __shfl_xor_sync(0xffffffff, bl, m);
      if (ov > bv || (ov == bv && ol < bl)) { bv = ov; bl = ol; }
    }
    // Exclude exactly the argmax lane, find second max.
    float v2 = (lane == bl) ? NEG_INF_F : v;
    float m2 = v2;
    #pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
      m2 = fmaxf(m2, __shfl_xor_sync(0xffffffff, m2, m));
    if (lane == 0) gs[warp] = bv + m2;
  }
  __syncthreads();

  // top-4 of 8 groups (warp 0)
  if (warp == 0) {
    float gv = (lane < N_GROUP) ? gs[lane] : NEG_INF_F;
    int   gi = lane;
    #pragma unroll
    for (int r = 0; r < TOPK_GROUP; r++) {
      float bv = gv;
      int   bi = gi;
      #pragma unroll
      for (int m = 16; m >= 1; m >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, bv, m);
        int   oi = __shfl_xor_sync(0xffffffff, bi, m);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      if (lane == 0) group_sel[bi] = 1;
      if (gi == bi) gv = NEG_INF_F;
    }
  }
  __syncthreads();

  // top-8 of E_GLOBAL among kept groups.
  float sp = (group_sel[tid / GROUP_SIZE] == 1) ? s_wb[tid] : NEG_INF_F;

  for (int r = 0; r < TOP_K; r++) {
    float bv = sp;
    int   bi = tid;
    // warp reduce
    #pragma unroll
    for (int m = 16; m >= 1; m >>= 1) {
      float ov = __shfl_xor_sync(0xffffffff, bv, m);
      int   oi = __shfl_xor_sync(0xffffffff, bi, m);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { warp_v[warp] = bv; warp_i[warp] = bi; }
    __syncthreads();
    if (warp == 0) {
      float bv2 = (lane < 8) ? warp_v[lane] : NEG_INF_F;
      int   bi2 = (lane < 8) ? warp_i[lane] : 0;
      #pragma unroll
      for (int m = 4; m >= 1; m >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, bv2, m);
        int   oi = __shfl_xor_sync(0xffffffff, bi2, m);
        if (ov > bv2 || (ov == bv2 && oi < bi2)) { bv2 = ov; bi2 = oi; }
      }
      if (lane == 0) {
        s_best_idx = bi2;
        tk_idx[r]  = bi2;
        tk_sig[r]  = s_sig[bi2];
      }
    }
    __syncthreads();
    if (tid == s_best_idx) sp = NEG_INF_F;
  }

  // normalize + scale
  if (tid == 0) {
    float ssum = 0.0f;
    #pragma unroll
    for (int k = 0; k < TOP_K; k++) ssum += tk_sig[k];
    float inv = 1.0f / (ssum + 1e-20f);
    #pragma unroll
    for (int k = 0; k < TOP_K; k++) {
      topk_idx_out[(size_t)t * TOP_K + k] = tk_idx[k];
      topk_w_out  [(size_t)t * TOP_K + k] = tk_sig[k] * inv * scaling_factor;
      int le = tk_idx[k] - local_expert_offset;
      if ((unsigned)le < (unsigned)E_LOCAL) {
        atomicAdd(&counts[le], 1);
      }
    }
  }
}

// ======================================================================
// Kernel 2: count tokens per local expert.
// ======================================================================
__global__ void perm_count_kernel(
    const int32_t* __restrict__ topk_idx,
    int T,
    int local_expert_offset,
    int32_t* __restrict__ counts)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  #pragma unroll
  for (int k = 0; k < TOP_K; k++) {
    int ge = topk_idx[(size_t)t * TOP_K + k];
    int le = ge - local_expert_offset;
    if ((unsigned)le < (unsigned)E_LOCAL) {
      atomicAdd(&counts[le], 1);
    }
  }
}

// ======================================================================
// Kernel 3: exclusive scan counts -> offs[E_LOCAL+1], write_ptr[E_LOCAL].
// ======================================================================
__global__ void scan_offsets_kernel(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ offs,
    int32_t* __restrict__ write_ptr,
    int32_t* __restrict__ n_sel_out)
{
  if (threadIdx.x != 0) return;
  int acc = 0;
  offs[0] = 0;
  #pragma unroll
  for (int i = 0; i < E_LOCAL; i++) {
    write_ptr[i] = acc;
    acc += counts[i];
    offs[i + 1] = acc;
  }
  *n_sel_out = acc;
}

// ======================================================================
// Kernel 4: fill perm arrays grouped by local expert.
// ======================================================================
__global__ void perm_fill_kernel(
    const int32_t* __restrict__ topk_idx,
    const float*   __restrict__ topk_w,
    int T,
    int local_expert_offset,
    int32_t* __restrict__ write_ptr,
    int32_t* __restrict__ perm_token,
    float*   __restrict__ perm_weight)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  #pragma unroll
  for (int k = 0; k < TOP_K; k++) {
    int ge = topk_idx[(size_t)t * TOP_K + k];
    int le = ge - local_expert_offset;
    if ((unsigned)le < (unsigned)E_LOCAL) {
      int slot = atomicAdd(&write_ptr[le], 1);
      perm_token [slot] = t;
      perm_weight[slot] = topk_w[(size_t)t * TOP_K + k];
    }
  }
}

// ======================================================================
// Kernel 5: hidden_states fp8 -> bf16.
// Scale[H/BS, T] -> row (hidden-block), col (token).
// Grid=(T, NH_BLKS), threads=BS. Block handles one (token, hb).
// ======================================================================
__global__ void dequant_hs_kernel(
    const __nv_fp8_e4m3* __restrict__ hs_fp8,
    const float*         __restrict__ hs_scale,
    int T,
    __nv_bfloat16* __restrict__ hs_bf16)
{
  const int t  = blockIdx.x;
  const int hb = blockIdx.y;
  const int d  = threadIdx.x;
  if (t >= T) return;
  float sc = hs_scale[(size_t)hb * T + t];
  int   h  = hb * BS + d;
  float v  = (float)hs_fp8[(size_t)t * H + h] * sc;
  hs_bf16[(size_t)t * H + h] = __float2bfloat16(v);
}

// ======================================================================
// Kernel 6: gather hs_bf16 rows into A_perm by perm_token.
// ======================================================================
__global__ void gather_hs_kernel(
    const __nv_bfloat16* __restrict__ hs_bf16,
    const int32_t*       __restrict__ perm_token,
    int N_sel,
    __nv_bfloat16* __restrict__ A_e)
{
  const int r  = blockIdx.x;
  const int hb = blockIdx.y;
  const int d  = threadIdx.x;
  if (r >= N_sel) return;
  int t = perm_token[r];
  int h = hb * BS + d;
  A_e[(size_t)r * H + h] = hs_bf16[(size_t)t * H + h];
}

// ======================================================================
// Kernel 7: weights fp8 -> bf16.
// Grid=(E_LOCAL, N/BS, K/BS), threads=256, 64 elems per thread.
// ======================================================================
__global__ void dequant_w_kernel(
    const __nv_fp8_e4m3* __restrict__ W_fp8,
    const float*         __restrict__ W_scale,
    const int32_t*       __restrict__ active_le,  // [A] local-expert ids to dequant
    int N, int K, int NB, int KB,
    __nv_bfloat16* __restrict__ W_bf16)
{
  const int le = active_le[blockIdx.x];
  const int nb = blockIdx.y;
  const int kb = blockIdx.z;
  const int tid = threadIdx.x;
  const int n0 = nb * BS;
  const int k0 = kb * BS;

  float sc = W_scale[((size_t)le * NB + nb) * KB + kb];

  constexpr int TILE  = BS * BS;       // 16384
  constexpr int ITERS = TILE / 256;    // 64
  #pragma unroll
  for (int it = 0; it < ITERS; it++) {
    int id = tid + it * 256;
    int r  = id / BS;
    int c  = id % BS;
    int n  = n0 + r;
    int k  = k0 + c;
    if (n < N && k < K) {
      size_t idx = ((size_t)le * N + n) * K + k;
      W_bf16[idx] = __float2bfloat16((float)W_fp8[idx] * sc);
    }
  }
}

// ======================================================================
// Kernel 8: bf16 GEMM, C[M,N] = A[M,K] @ B[N,K]^T, row-major.
// BM=128, BN=128, BK=32. 8 warps laid out 4 (M) x 2 (N).
// Each warp: 32 x 64 output = 2 x 4 WMMA 16x16x16.
// 2-stage cp.async pipeline.
// ======================================================================
constexpr int BM           = 128;
constexpr int BN           = 128;
constexpr int BK           = 32;
constexpr int NWARPS_M     = 4;
constexpr int NWARPS_N     = 2;
constexpr int NWARPS       = NWARPS_M * NWARPS_N;   // 8
constexpr int THREADS_GEMM = NWARPS * 32;           // 256
constexpr int WM           = BM / NWARPS_M;         // 32
constexpr int WN           = BN / NWARPS_N;         // 64
constexpr int W_TILES_M    = WM / 16;               // 2
constexpr int W_TILES_N    = WN / 16;               // 4
constexpr int LDA_S        = BK + 8;                // 40 (bf16 pad)
constexpr int LDB_S        = BK + 8;                // 40
constexpr int STAGES       = 2;

static_assert(BM * BK % (THREADS_GEMM * 8) == 0, "A load not evenly tileable by 16B");
static_assert(BN * BK % (THREADS_GEMM * 8) == 0, "B load not evenly tileable by 16B");

__device__ __forceinline__ void cp_async_16(void* sdst, const void* gsrc, bool pred) {
  unsigned sdstaddr = __cvta_generic_to_shared(sdst);
  if (pred) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(sdstaddr), "l"(gsrc));
  } else {
    *reinterpret_cast<uint4*>(sdst) = make_uint4(0, 0, 0, 0);
  }
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void stage_load_A(
    __nv_bfloat16 (*sA)[LDA_S],
    const __nv_bfloat16* A, int m0,
    int k0, int M, int K, int tid)
{
  constexpr int A_ITERS = (BM * BK) / (THREADS_GEMM * 8);
  #pragma unroll
  for (int it = 0; it < A_ITERS; it++) {
    int id = tid + it * THREADS_GEMM;
    int r  = id / (BK / 8);
    int c  = (id % (BK / 8)) * 8;
    bool ok = (m0 + r) < M;
    const void* gptr = &A[(size_t)(m0 + r) * K + (k0 + c)];
    cp_async_16(&sA[r][c], gptr, ok);
  }
}

__device__ __forceinline__ void stage_make_swiglu_A(
    __nv_bfloat16 (*sA)[LDA_S],
    const float* G1, int m0,
    int k0, int M, int tid)
{
  constexpr int A_ITERS = (BM * BK) / (THREADS_GEMM * 8);
  #pragma unroll
  for (int it = 0; it < A_ITERS; it++) {
    int id = tid + it * THREADS_GEMM;
    int r  = id / (BK / 8);
    int c  = (id % (BK / 8)) * 8;
    int gm = m0 + r;
    int gk = k0 + c;
    __nv_bfloat16 dst[8];
    if (gm < M) {
      const float* row = G1 + (size_t)gm * II;
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        float x1 = row[gk + i];
        float x2 = row[I + gk + i];
        float silu = x2 / (1.0f + __expf(-x2));
        dst[i] = __float2bfloat16(silu * x1);
      }
    } else {
      #pragma unroll
      for (int i = 0; i < 8; i++) dst[i] = __float2bfloat16(0.0f);
    }
    *reinterpret_cast<uint4*>(&sA[r][c]) = *reinterpret_cast<const uint4*>(dst);
  }
}

__device__ __forceinline__ void stage_load_B_bf16(
    __nv_bfloat16 (*sB)[LDB_S],
    const __nv_bfloat16* B, int n0,
    int k0, int N, int K, int tid)
{
  constexpr int B_ITERS = (BN * BK) / (THREADS_GEMM * 8);
  #pragma unroll
  for (int it = 0; it < B_ITERS; it++) {
    int id = tid + it * THREADS_GEMM;
    int r  = id / (BK / 8);
    int c  = (id % (BK / 8)) * 8;
    bool ok = (n0 + r) < N;
    const void* gptr = &B[(size_t)(n0 + r) * K + (k0 + c)];
    cp_async_16(&sB[r][c], gptr, ok);
  }
}

// LDB_S padded stride for the bf16 sB buffer (kept in bytes the same as LDB_S).
constexpr int LDB_FP8 = BK + 16;       // fp8 pad (16 elems = 16 bytes)

// Async FP8 load: cp.async 16B of fp8 per thread into a scratch buffer.
// No conversion happens here — fp8 stays as raw bytes in shmem.
__device__ __forceinline__ void stage_load_B_fp8_async(
    __nv_fp8_e4m3 (*sB_fp8)[LDB_FP8],
    const __nv_fp8_e4m3* B_fp8,
    int n0, int k0, int N, int K, int tid)
{
  constexpr int B_ITERS = (BN * BK) / (THREADS_GEMM * 16);
  #pragma unroll
  for (int it = 0; it < B_ITERS; it++) {
    int id = tid + it * THREADS_GEMM;
    int r  = id / (BK / 16);          // 0..127 (since BK/16 = 2)
    int c  = (id % (BK / 16)) * 16;    // 0, 16
    bool ok = (n0 + r) < N;
    const void* gptr = &B_fp8[(size_t)(n0 + r) * K + (k0 + c)];
    cp_async_16(&sB_fp8[r][c], gptr, ok);
  }
}

// Convert sB_fp8 scratch -> sB bf16 with fused per-tile scale multiply.
// Called once per main-loop iter after async loads complete.
__device__ __forceinline__ void convert_sB_fp8_to_bf16(
    const __nv_fp8_e4m3 (*sB_fp8)[LDB_FP8],
    __nv_bfloat16 (*sB)[LDB_S],
    float sc, int tid)
{
  constexpr int ITERS = (BN * BK) / (THREADS_GEMM * 16);
  #pragma unroll
  for (int it = 0; it < ITERS; it++) {
    int id = tid + it * THREADS_GEMM;
    int r  = id / (BK / 16);
    int c  = (id % (BK / 16)) * 16;
    uint4 u = *reinterpret_cast<const uint4*>(&sB_fp8[r][c]);
    __nv_fp8_e4m3 b[16];
    *reinterpret_cast<uint4*>(b) = u;
    __nv_bfloat16 dst[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) dst[i] = __float2bfloat16((float)b[i] * sc);
    reinterpret_cast<uint4*>(&sB[r][c])[0] = reinterpret_cast<const uint4*>(dst)[0];
    reinterpret_cast<uint4*>(&sB[r][c])[1] = reinterpret_cast<const uint4*>(dst)[1];
  }
}

// Dynamic shared memory layout (in bytes, offset from smem_raw):
//   [0            , SA_BYTES)          : sA[STAGES][BM][LDA_S] bf16
//   [SA_BYTES     , SA_BYTES+SB_BYTES) : sB[STAGES][BN][LDB_S] bf16
// Epilogue reuses same buffer (main loop done):
//   [0, SC_BYTES)                      : sW[NWARPS][16][16] fp32 (per-warp 1KB)
constexpr size_t SA_BYTES = (size_t)STAGES * BM * LDA_S * sizeof(__nv_bfloat16);
constexpr size_t SB_BYTES = (size_t)STAGES * BN * LDB_S * sizeof(__nv_bfloat16);
constexpr size_t SC_BYTES = (size_t)NWARPS * 16 * 16 * sizeof(float);
constexpr size_t GEMM_SMEM_BYTES =
    (SA_BYTES + SB_BYTES) > SC_BYTES ? (SA_BYTES + SB_BYTES) : SC_BYTES;

__global__ void gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    int M, int N, int K,
    __nv_bfloat16* __restrict__ C,
    int ldC)
{
  const int tid    = threadIdx.x;
  const int warp   = tid >> 5;
  const int warp_m = warp / NWARPS_N;
  const int warp_n = warp % NWARPS_N;

  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16 (*sA)[BM][LDA_S] =
      reinterpret_cast<__nv_bfloat16 (*)[BM][LDA_S]>(smem_raw);
  __nv_bfloat16 (*sB)[BN][LDB_S] =
      reinterpret_cast<__nv_bfloat16 (*)[BN][LDB_S]>(smem_raw + SA_BYTES);

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>    fA[W_TILES_M];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>    fB[W_TILES_N];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>                          fC[W_TILES_M][W_TILES_N];
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N; j++)
      wmma::fill_fragment(fC[i][j], 0.0f);

  stage_load_A(sA[0], A, m0, 0, M, K, tid);
  stage_load_B_bf16(sB[0], B, n0, 0, N, K, tid);
  cp_async_commit();

  int stage = 0;
  for (int k = 0; k < K; k += BK) {
    int next_k = k + BK;
    if (next_k < K) {
      int next_stage = 1 - stage;
      stage_load_A(sA[next_stage], A, m0, next_k, M, K, tid);
      stage_load_B_bf16(sB[next_stage], B, n0, next_k, N, K, tid);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++)
        wmma::load_matrix_sync(fA[i], &sA[stage][warp_m * WM + i * 16][kk], LDA_S);
      #pragma unroll
      for (int j = 0; j < W_TILES_N; j++)
        wmma::load_matrix_sync(fB[j], &sB[stage][warp_n * WN + j * 16][kk], LDB_S);
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < W_TILES_N; j++)
          wmma::mma_sync(fC[i][j], fA[i], fB[j], fC[i][j]);
    }

    stage = 1 - stage;
  }

  // Main loop complete — reuse smem as fp32 accumulator buffer.
  // Per-warp epilogue: one 16x16 shmem scratch per warp keeps total epilogue
  // smem at 8*1KB = 8KB (vs 64KB for a full BM×BN fp32 buffer).
  const int lane = tid & 31;
  __syncthreads();
  float (*sW)[16][16] = reinterpret_cast<float (*)[16][16]>(smem_raw);
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N; j++) {
      wmma::store_matrix_sync(&sW[warp][0][0], fC[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      #pragma unroll
      for (int e = lane; e < 256; e += 32) {
        int r = e >> 4;
        int c = e & 15;
        int gm = m0 + warp_m * WM + i * 16 + r;
        int gn = n0 + warp_n * WN + j * 16 + c;
        if (gm < M && gn < N) {
          C[(size_t)gm * ldC + gn] = __float2bfloat16(sW[warp][r][c]);
        }
      }
      __syncwarp();
    }
}

// Fused FP8-weight GEMM: A bf16, B fp8 + scale.
// B is async-loaded as fp8 via cp.async to scratch, then converted in-shmem to bf16.
constexpr size_t SA_BYTES_V2  = (size_t)STAGES * BM * LDA_S    * sizeof(__nv_bfloat16);
constexpr size_t SBFP8_BYTES  = (size_t)STAGES * BN * LDB_FP8  * sizeof(__nv_fp8_e4m3);
constexpr size_t SBBF16_BYTES = (size_t)       BN * LDB_S      * sizeof(__nv_bfloat16);
constexpr size_t MAIN_BYTES_V2 = SA_BYTES_V2 + SBFP8_BYTES + SBBF16_BYTES;
constexpr size_t SC_BYTES_V2   = (size_t)NWARPS * 16 * 16 * sizeof(float);
constexpr size_t GEMM_SMEM_BYTES_V2 =
    MAIN_BYTES_V2 > SC_BYTES_V2 ? MAIN_BYTES_V2 : SC_BYTES_V2;

__global__ void gemm_bf16_fp8w_kernel(
    const __nv_bfloat16* __restrict__ A,   // [M, K]
    const __nv_fp8_e4m3* __restrict__ B_fp8,  // [N, K]
    const float*         __restrict__ B_scale,// [NB, KB]
    int M, int N, int K, int NB, int KB,
    __nv_bfloat16* __restrict__ C,
    int ldC)
{
  const int tid    = threadIdx.x;
  const int warp   = tid >> 5;
  const int warp_m = warp / NWARPS_N;
  const int warp_n = warp % NWARPS_N;

  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int nb = n0 / BS;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16 (*sA)[BM][LDA_S] =
      reinterpret_cast<__nv_bfloat16 (*)[BM][LDA_S]>(smem_raw);
  __nv_fp8_e4m3 (*sB_fp8)[BN][LDB_FP8] =
      reinterpret_cast<__nv_fp8_e4m3 (*)[BN][LDB_FP8]>(smem_raw + SA_BYTES_V2);
  __nv_bfloat16 (*sB)[LDB_S] =
      reinterpret_cast<__nv_bfloat16 (*)[LDB_S]>(
          smem_raw + SA_BYTES_V2 + SBFP8_BYTES);

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>    fA[W_TILES_M];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>    fB[W_TILES_N];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>                          fC[W_TILES_M][W_TILES_N];
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N; j++)
      wmma::fill_fragment(fC[i][j], 0.0f);

  // Preload stage 0: A bf16 + B fp8 (both via cp.async).
  stage_load_A(sA[0], A, m0, 0, M, K, tid);
  stage_load_B_fp8_async(sB_fp8[0], B_fp8, n0, 0, N, K, tid);
  cp_async_commit();

  int stage = 0;
  for (int k = 0; k < K; k += BK) {
    int next_k = k + BK;
    if (next_k < K) {
      int next_stage = 1 - stage;
      stage_load_A(sA[next_stage], A, m0, next_k, M, K, tid);
      stage_load_B_fp8_async(sB_fp8[next_stage], B_fp8, n0, next_k, N, K, tid);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    // Convert current stage's fp8 to bf16 with per-tile scale.
    int kb = k / BS;
    float sc = B_scale[(size_t)nb * KB + kb];
    convert_sB_fp8_to_bf16(sB_fp8[stage], sB, sc, tid);
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++)
        wmma::load_matrix_sync(fA[i], &sA[stage][warp_m * WM + i * 16][kk], LDA_S);
      #pragma unroll
      for (int j = 0; j < W_TILES_N; j++)
        wmma::load_matrix_sync(fB[j], &sB[warp_n * WN + j * 16][kk], LDB_S);
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < W_TILES_N; j++)
          wmma::mma_sync(fC[i][j], fA[i], fB[j], fC[i][j]);
    }

    stage = 1 - stage;
  }

  const int lane = tid & 31;
  __syncthreads();
  float (*sW)[16][16] = reinterpret_cast<float (*)[16][16]>(smem_raw);
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N; j++) {
      wmma::store_matrix_sync(&sW[warp][0][0], fC[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      #pragma unroll
      for (int e = lane; e < 256; e += 32) {
        int r = e >> 4;
        int c = e & 15;
        int gm = m0 + warp_m * WM + i * 16 + r;
        int gn = n0 + warp_n * WN + j * 16 + c;
        if (gm < M && gn < N) {
          C[(size_t)gm * ldC + gn] = __float2bfloat16(sW[warp][r][c]);
        }
      }
      __syncwarp();
    }
}

// ======================================================================
// Grouped GEMM: one persistent kernel covers all active experts.
// Each block reads its (expert, tile_m) from a schedule array; tile_n = blockIdx.x.
// 2-stage cp.async pipeline (same as per-expert kernel).
// ======================================================================
constexpr size_t GEMM_SMEM_BYTES_GRP = GEMM_SMEM_BYTES_V2;

__global__ void gemm_bf16_fp8w_grouped_kernel(
    const __nv_bfloat16* __restrict__ A_base,     // [N_sel, K]
    const __nv_fp8_e4m3* __restrict__ W_base,     // [E_LOCAL, N, K]
    const float*         __restrict__ S_base,     // [E_LOCAL, NB, KB]
    const int32_t*       __restrict__ offs,       // [E_LOCAL+1]
    const int32_t*       __restrict__ sched_e,    // [total_m_tiles]
    const int32_t*       __restrict__ sched_tm,   // [total_m_tiles]
    int N, int K, int NB, int KB,
    __nv_bfloat16* __restrict__ C_base,           // [N_sel, ldC]
    int ldC)
{
  const int e    = sched_e[blockIdx.y];
  const int tm   = sched_tm[blockIdx.y];
  const int rs   = offs[e];
  const int re   = offs[e + 1];
  const int M    = re - rs;
  const int m0   = tm * BM;
  const int n0   = blockIdx.x * BN;
  if (m0 >= M) return;

  const __nv_bfloat16* A       = A_base + (size_t)rs * K;
  const __nv_fp8_e4m3* B_fp8   = W_base + (size_t)e * (size_t)N * (size_t)K;
  const float*         B_scale = S_base + (size_t)e * (size_t)NB * (size_t)KB;
  __nv_bfloat16*       C       = C_base + (size_t)rs * (size_t)ldC;

  const int tid    = threadIdx.x;
  const int warp   = tid >> 5;
  const int warp_m = warp / NWARPS_N;
  const int warp_n = warp % NWARPS_N;
  const int nb     = n0 / BS;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16 (*sA)[BM][LDA_S] =
      reinterpret_cast<__nv_bfloat16 (*)[BM][LDA_S]>(smem_raw);
  __nv_fp8_e4m3 (*sB_fp8)[BN][LDB_FP8] =
      reinterpret_cast<__nv_fp8_e4m3 (*)[BN][LDB_FP8]>(smem_raw + SA_BYTES_V2);
  __nv_bfloat16 (*sB)[LDB_S] =
      reinterpret_cast<__nv_bfloat16 (*)[LDB_S]>(
          smem_raw + SA_BYTES_V2 + SBFP8_BYTES);

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>    fA[W_TILES_M];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>    fB[W_TILES_N];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>                          fC[W_TILES_M][W_TILES_N];
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N; j++)
      wmma::fill_fragment(fC[i][j], 0.0f);

  stage_load_A(sA[0], A, m0, 0, M, K, tid);
  stage_load_B_fp8_async(sB_fp8[0], B_fp8, n0, 0, N, K, tid);
  cp_async_commit();

  int stage = 0;
  for (int k = 0; k < K; k += BK) {
    int next_k = k + BK;
    if (next_k < K) {
      int next_stage = 1 - stage;
      stage_load_A(sA[next_stage], A, m0, next_k, M, K, tid);
      stage_load_B_fp8_async(sB_fp8[next_stage], B_fp8, n0, next_k, N, K, tid);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    int kb = k / BS;
    float sc = B_scale[(size_t)nb * KB + kb];
    convert_sB_fp8_to_bf16(sB_fp8[stage], sB, sc, tid);
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++)
        wmma::load_matrix_sync(fA[i], &sA[stage][warp_m * WM + i * 16][kk], LDA_S);
      #pragma unroll
      for (int j = 0; j < W_TILES_N; j++)
        wmma::load_matrix_sync(fB[j], &sB[warp_n * WN + j * 16][kk], LDB_S);
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < W_TILES_N; j++)
          wmma::mma_sync(fC[i][j], fA[i], fB[j], fC[i][j]);
    }

    stage = 1 - stage;
  }

  const int lane = tid & 31;
  __syncthreads();
  float (*sW)[16][16] = reinterpret_cast<float (*)[16][16]>(smem_raw);
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N; j++) {
      wmma::store_matrix_sync(&sW[warp][0][0], fC[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      #pragma unroll
      for (int ee = lane; ee < 256; ee += 32) {
        int r = ee >> 4;
        int c = ee & 15;
        int gm = m0 + warp_m * WM + i * 16 + r;
        int gn = n0 + warp_n * WN + j * 16 + c;
        if (gm < M && gn < N) {
          C[(size_t)gm * ldC + gn] = __float2bfloat16(sW[warp][r][c]);
        }
      }
      __syncwarp();
    }
}

// ======================================================================
// Grouped BF16 GEMM with FP8 weights + fused fin_acc:
//   scatters weighted fp32 output directly to per-token accumulator,
//   skipping the bf16 O intermediate that caused ~1 bf16-LSB precision loss
//   per element (~2048 at output magnitudes of 5e5 seen in random workloads).
// ======================================================================
__global__ void gemm_bf16_fp8w_fused_acc_grouped_kernel(
    const __nv_bfloat16* __restrict__ A_base,
    const __nv_fp8_e4m3* __restrict__ W_base,
    const float*         __restrict__ S_base,
    const int32_t*       __restrict__ offs,
    const int32_t*       __restrict__ sched_e,
    const int32_t*       __restrict__ sched_tm,
    const int32_t*       __restrict__ perm_token,
    const float*         __restrict__ perm_weight,
    int N, int K, int NB, int KB,
    float* __restrict__ acc_out,
    int ldAcc)
{
  const int e    = sched_e[blockIdx.y];
  const int tm   = sched_tm[blockIdx.y];
  const int rs   = offs[e];
  const int re   = offs[e + 1];
  const int M    = re - rs;
  const int m0   = tm * BM;
  const int n0   = blockIdx.x * BN;
  if (m0 >= M) return;

  const __nv_bfloat16* A       = A_base + (size_t)rs * K;
  const __nv_fp8_e4m3* B_fp8   = W_base + (size_t)e * (size_t)N * (size_t)K;
  const float*         B_scale = S_base + (size_t)e * (size_t)NB * (size_t)KB;
  const int32_t*       Ptok    = perm_token + rs;
  const float*         Pwgt    = perm_weight + rs;

  const int tid    = threadIdx.x;
  const int warp   = tid >> 5;
  const int warp_m = warp / NWARPS_N;
  const int warp_n = warp % NWARPS_N;
  const int nb     = n0 / BS;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16 (*sA)[BM][LDA_S] =
      reinterpret_cast<__nv_bfloat16 (*)[BM][LDA_S]>(smem_raw);
  __nv_fp8_e4m3 (*sB_fp8)[BN][LDB_FP8] =
      reinterpret_cast<__nv_fp8_e4m3 (*)[BN][LDB_FP8]>(smem_raw + SA_BYTES_V2);
  __nv_bfloat16 (*sB)[LDB_S] =
      reinterpret_cast<__nv_bfloat16 (*)[LDB_S]>(
          smem_raw + SA_BYTES_V2 + SBFP8_BYTES);

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>    fA[W_TILES_M];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>    fB[W_TILES_N];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>                          fC[W_TILES_M][W_TILES_N];
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N; j++)
      wmma::fill_fragment(fC[i][j], 0.0f);

  stage_load_A(sA[0], A, m0, 0, M, K, tid);
  stage_load_B_fp8_async(sB_fp8[0], B_fp8, n0, 0, N, K, tid);
  cp_async_commit();

  int stage = 0;
  for (int k = 0; k < K; k += BK) {
    int next_k = k + BK;
    if (next_k < K) {
      int next_stage = 1 - stage;
      stage_load_A(sA[next_stage], A, m0, next_k, M, K, tid);
      stage_load_B_fp8_async(sB_fp8[next_stage], B_fp8, n0, next_k, N, K, tid);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    int kb = k / BS;
    float sc = B_scale[(size_t)nb * KB + kb];
    convert_sB_fp8_to_bf16(sB_fp8[stage], sB, sc, tid);
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++)
        wmma::load_matrix_sync(fA[i], &sA[stage][warp_m * WM + i * 16][kk], LDA_S);
      #pragma unroll
      for (int j = 0; j < W_TILES_N; j++)
        wmma::load_matrix_sync(fB[j], &sB[warp_n * WN + j * 16][kk], LDB_S);
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < W_TILES_N; j++)
          wmma::mma_sync(fC[i][j], fA[i], fB[j], fC[i][j]);
    }

    stage = 1 - stage;
  }

  // Fused epilogue: write fp32 × per-row weight atomically to acc_out[token, col].
  const int lane = tid & 31;
  __syncthreads();
  float (*sW)[16][16] = reinterpret_cast<float (*)[16][16]>(smem_raw);
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N; j++) {
      wmma::store_matrix_sync(&sW[warp][0][0], fC[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      #pragma unroll
      for (int ee = lane; ee < 256; ee += 32) {
        int r = ee >> 4;
        int c = ee & 15;
        int gm = m0 + warp_m * WM + i * 16 + r;
        int gn = n0 + warp_n * WN + j * 16 + c;
        if (gm < M && gn < N) {
          int t = Ptok[gm];
          float w = Pwgt[gm];
          float v = sW[warp][r][c] * w;
          atomicAdd(&acc_out[(size_t)t * ldAcc + gn], v);
        }
      }
      __syncwarp();
	    }
}

// GEMM2 fused with SwiGLU input staging:
//   A tile is computed from fp32 G1 as bf16(silu(gate) * up) directly in smem,
//   then multiplied by fp8 W2 and scattered to the final fp32 accumulator.
__global__ void gemm_swiglu_fp8w_fused_acc_grouped_kernel(
    const float*        __restrict__ G1_base,     // [N_sel, 2I]
    const __nv_fp8_e4m3* __restrict__ W_base,
    const float*         __restrict__ S_base,
    const int32_t*       __restrict__ offs,
    const int32_t*       __restrict__ sched_e,
    const int32_t*       __restrict__ sched_tm,
    const int32_t*       __restrict__ perm_token,
    const float*         __restrict__ perm_weight,
    int N, int K, int NB, int KB,
    float* __restrict__ acc_out,
    int ldAcc)
{
  const int e    = sched_e[blockIdx.y];
  const int tm   = sched_tm[blockIdx.y];
  const int rs   = offs[e];
  const int re   = offs[e + 1];
  const int M    = re - rs;
  const int m0   = tm * BM;
  const int n0   = blockIdx.x * BN;
  if (m0 >= M) return;

  const float*         G1      = G1_base + (size_t)rs * II;
  const __nv_fp8_e4m3* B_fp8   = W_base + (size_t)e * (size_t)N * (size_t)K;
  const float*         B_scale = S_base + (size_t)e * (size_t)NB * (size_t)KB;
  const int32_t*       Ptok    = perm_token + rs;
  const float*         Pwgt    = perm_weight + rs;

  const int tid    = threadIdx.x;
  const int warp   = tid >> 5;
  const int warp_m = warp / NWARPS_N;
  const int warp_n = warp % NWARPS_N;
  const int nb     = n0 / BS;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16 (*sA)[BM][LDA_S] =
      reinterpret_cast<__nv_bfloat16 (*)[BM][LDA_S]>(smem_raw);
  __nv_fp8_e4m3 (*sB_fp8)[BN][LDB_FP8] =
      reinterpret_cast<__nv_fp8_e4m3 (*)[BN][LDB_FP8]>(smem_raw + SA_BYTES_V2);
  __nv_bfloat16 (*sB)[LDB_S] =
      reinterpret_cast<__nv_bfloat16 (*)[LDB_S]>(
          smem_raw + SA_BYTES_V2 + SBFP8_BYTES);

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>    fA[W_TILES_M];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>    fB[W_TILES_N];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>                          fC[W_TILES_M][W_TILES_N];
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N; j++)
      wmma::fill_fragment(fC[i][j], 0.0f);

  stage_make_swiglu_A(sA[0], G1, m0, 0, M, tid);
  stage_load_B_fp8_async(sB_fp8[0], B_fp8, n0, 0, N, K, tid);
  cp_async_commit();

  int stage = 0;
  for (int k = 0; k < K; k += BK) {
    int next_k = k + BK;
    if (next_k < K) {
      int next_stage = 1 - stage;
      stage_make_swiglu_A(sA[next_stage], G1, m0, next_k, M, tid);
      stage_load_B_fp8_async(sB_fp8[next_stage], B_fp8, n0, next_k, N, K, tid);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    int kb = k / BS;
    float sc = B_scale[(size_t)nb * KB + kb];
    convert_sB_fp8_to_bf16(sB_fp8[stage], sB, sc, tid);
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++)
        wmma::load_matrix_sync(fA[i], &sA[stage][warp_m * WM + i * 16][kk], LDA_S);
      #pragma unroll
      for (int j = 0; j < W_TILES_N; j++)
        wmma::load_matrix_sync(fB[j], &sB[warp_n * WN + j * 16][kk], LDB_S);
      #pragma unroll
      for (int i = 0; i < W_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < W_TILES_N; j++)
          wmma::mma_sync(fC[i][j], fA[i], fB[j], fC[i][j]);
    }

    stage = 1 - stage;
  }

  const int lane = tid & 31;
  __syncthreads();
  float (*sW)[16][16] = reinterpret_cast<float (*)[16][16]>(smem_raw);
  #pragma unroll
  for (int i = 0; i < W_TILES_M; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N; j++) {
      wmma::store_matrix_sync(&sW[warp][0][0], fC[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      #pragma unroll
      for (int ee = lane; ee < 256; ee += 32) {
        int r = ee >> 4;
        int c = ee & 15;
        int gm = m0 + warp_m * WM + i * 16 + r;
        int gn = n0 + warp_n * WN + j * 16 + c;
        if (gm < M && gn < N) {
          int t = Ptok[gm];
          float w = Pwgt[gm];
          float v = sW[warp][r][c] * w;
          atomicAdd(&acc_out[(size_t)t * ldAcc + gn], v);
        }
      }
      __syncwarp();
    }
}

// ======================================================================
// FP8 MMA grouped GEMM1: A fp8 [N_sel, K] × W fp8 [E, N, K]^T → bf16 [N_sel, N]
// A scale is per-row per-kblock: A_scale[N_sel, KB]
// W scale is per-nblock per-kblock: W_scale[E, NB, KB]
// BK = 128 (one kblock per BK iter, so block-scale apply happens at BK boundary).
// ======================================================================
constexpr int BK_FP8          = 128;
constexpr int LDA_FP8_PAD     = BK_FP8 + 16;      // row stride in fp8 bytes
constexpr int LDB_FP8_PAD     = BK_FP8 + 16;
constexpr int STAGES_FP8      = 2;
// FP8 kernel layout: 8 warps (NWARPS_M_FP8=4 × NWARPS_N_FP8=2), WM=32 WN=64,
// 2-stage cp.async pipeline. Each warp owns 2×8 = 16 m16n8k32 sub-tiles,
// and the kernel ends up at ~255 regs/thread → 1 block/SM = 12.5% achieved
// occupancy. Experiments (__launch_bounds__ forcing 2 blocks/SM, 16-warp
// 32×32 tile, 3-stage pipeline) all regressed ~3-4% on Modal — reducing
// register pressure pushes the kernel from compute-bound (51% SM) to
// shmem/DRAM-bound, and the extra prologue of a 3-stage pipeline doesn't
// amortize over K=7168. The 255-reg / 12.5%-occ / 2-stage point is the
// sweet spot here.
constexpr int NWARPS_M_FP8    = 4;
constexpr int NWARPS_N_FP8    = 2;
constexpr int NWARPS_FP8      = NWARPS_M_FP8 * NWARPS_N_FP8;      // 8
constexpr int THREADS_FP8     = NWARPS_FP8 * 32;                   // 256
constexpr int WM_FP8          = BM / NWARPS_M_FP8;                 // 32
constexpr int WN_FP8          = BN / NWARPS_N_FP8;                 // 64
constexpr int W_TILES_M_FP8   = WM_FP8 / 16;                       // 2
constexpr int WN_TILES_FP8    = WN_FP8 / 8;                        // 8
constexpr size_t SA_FP8_BYTES = (size_t)STAGES_FP8 * BM * LDA_FP8_PAD;  // fp8 bytes
constexpr size_t SB_FP8_BYTES = (size_t)STAGES_FP8 * BN * LDB_FP8_PAD;

// bf16 upcast region for the Triton-precision kernel.
// fp8 load buffers are 3-stage (2 kblocks of prefetch-ahead to hide HBM latency);
// bf16 conversion buffers are single-stage (consumed immediately by the mma loop).
constexpr int LDA_BF16_PAD = BK_FP8 + 8;        // 136 bf16 elements/row
constexpr int LDB_BF16_PAD = BK_FP8 + 8;
constexpr int GEMM1_STAGES = 3;
constexpr size_t SA_FP8_STG_BYTES = (size_t)GEMM1_STAGES * BM * LDA_FP8_PAD;                 // 55296
constexpr size_t SB_FP8_STG_BYTES = (size_t)GEMM1_STAGES * BN * LDB_FP8_PAD;                 // 55296
constexpr size_t SA_BF16_BYTES    = (size_t)BM * LDA_BF16_PAD * sizeof(__nv_bfloat16);       // 34816
constexpr size_t SB_BF16_BYTES    = (size_t)BN * LDB_BF16_PAD * sizeof(__nv_bfloat16);
constexpr size_t GEMM_FP8_SMEM_BYTES = SA_FP8_STG_BYTES + SB_FP8_STG_BYTES
                                     + SA_BF16_BYTES + SB_BF16_BYTES;  // ~176 KB

__device__ __forceinline__ void mma_m16n8k32_e4m3(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1));
}

// ldmatrix.x4.b16 was tried here to replace the 4 manual u32 loads per
// thread in the hot kk loop. Result: GEMM1 regressed 6.23ms → 6.49ms
// (+4%) despite halving the explicit load count. The compiler was
// already coalescing the 4 consecutive u32 loads (same thread, same
// base) into one LDS.128, so ldmatrix doesn't reduce load bandwidth;
// it just adds address-compute overhead for the 32 per-lane row
// pointers. Keeping the manual-load form until a bigger rewrite
// (producer-consumer or tcgen05) changes the access pattern.

// ======================================================================
// mbarrier PTX helpers for warp-specialized producer/consumer pipelining.
// Each stage has a `ready` barrier (producer → consumer) and a `consumed`
// barrier (consumer → producer). Barriers live in shared memory and are
// initialized once at kernel entry.
// ======================================================================
__device__ __forceinline__ void mbarrier_init_shared(uint64_t* bar, uint32_t count) {
  unsigned addr = __cvta_generic_to_shared(bar);
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_shared(uint64_t* bar) {
  unsigned addr = __cvta_generic_to_shared(bar);
  uint64_t state;
  asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];"
               : "=l"(state) : "r"(addr));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx_shared(uint64_t* bar, uint32_t tx_count) {
  unsigned addr = __cvta_generic_to_shared(bar);
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
               :: "r"(addr), "r"(tx_count));
}

__device__ __forceinline__ void mbarrier_wait_parity_shared(uint64_t* bar, uint32_t phase) {
  unsigned addr = __cvta_generic_to_shared(bar);
  asm volatile(
    "{ .reg .pred p; waitLoop:\n"
    "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
    "  @!p bra waitLoop; }\n"
    :: "r"(addr), "r"(phase));
}

// Delayed arrive: this thread's arrival on `bar` is pending until its
// outstanding cp.async group completes. Lets producers issue loads and
// move on without a block-global cp_async_wait<0>. PTX op takes only
// the mbar address (no .shared qualifier — mbarrier is implicitly shmem).
__device__ __forceinline__ void cp_async_mbarrier_arrive(uint64_t* bar) {
  unsigned addr = __cvta_generic_to_shared(bar);
  asm volatile("cp.async.mbarrier.arrive.b64 [%0];" :: "r"(addr));
}

__device__ __forceinline__ void cp_async_bulk_1d_shared_global(
    void* sdst, const void* gsrc, uint32_t bytes, uint64_t* bar) {
  unsigned sdstaddr = __cvta_generic_to_shared(sdst);
  unsigned mbaraddr = __cvta_generic_to_shared(bar);
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];"
      :: "r"(sdstaddr), "l"(gsrc), "r"(bytes), "r"(mbaraddr) : "memory");
}

__device__ __forceinline__ void cp_async_bulk_tensor_2d_shared_global(
    void* sdst, const CUtensorMap* tmap, int c0, int c1, uint64_t* bar) {
  unsigned sdstaddr = __cvta_generic_to_shared(sdst);
  unsigned mbaraddr = __cvta_generic_to_shared(bar);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.tile "
      "[%0], [%1, {%2, %3}], [%4];"
      :: "r"(sdstaddr), "l"(tmap), "r"(c0), "r"(c1), "r"(mbaraddr) : "memory");
}

__device__ __forceinline__ void cp_async_bulk_tensor_3d_shared_global(
    void* sdst, const CUtensorMap* tmap, int c0, int c1, int c2, uint64_t* bar) {
  unsigned sdstaddr = __cvta_generic_to_shared(sdst);
  unsigned mbaraddr = __cvta_generic_to_shared(bar);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.tile "
      "[%0], [%1, {%2, %3, %4}], [%5];"
      :: "r"(sdstaddr), "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(mbaraddr) : "memory");
}

template <int THREADS>
__device__ __forceinline__ void stage_load_A_fp8_BK128(
    __nv_fp8_e4m3 (*sA)[LDA_FP8_PAD],
    const __nv_fp8_e4m3* A, int m0, int k0, int M, int K, int tid)
{
  constexpr int A_ITERS = (BM * BK_FP8) / (THREADS * 16);
  static_assert((BM * BK_FP8) % (THREADS * 16) == 0, "A load not tileable");
  #pragma unroll
  for (int it = 0; it < A_ITERS; it++) {
    int id = tid + it * THREADS;
    int r  = id / (BK_FP8 / 16);
    int c  = (id % (BK_FP8 / 16)) * 16;
    bool ok = (m0 + r) < M;
    const void* gptr = &A[(size_t)(m0 + r) * K + (k0 + c)];
    cp_async_16(&sA[r][c], gptr, ok);
  }
}

template <int THREADS>
__device__ __forceinline__ void stage_load_B_fp8_BK128(
    __nv_fp8_e4m3 (*sB)[LDB_FP8_PAD],
    const __nv_fp8_e4m3* B, int n0, int k0, int N, int K, int tid)
{
  constexpr int B_ITERS = (BN * BK_FP8) / (THREADS * 16);
  static_assert((BN * BK_FP8) % (THREADS * 16) == 0, "B load not tileable");
  #pragma unroll
  for (int it = 0; it < B_ITERS; it++) {
    int id = tid + it * THREADS;
    int r  = id / (BK_FP8 / 16);
    int c  = (id % (BK_FP8 / 16)) * 16;
    bool ok = (n0 + r) < N;
    const void* gptr = &B[(size_t)(n0 + r) * K + (k0 + c)];
    cp_async_16(&sB[r][c], gptr, ok);
  }
}

// GEMM1 grouped kernel, Triton-precision variant:
//   fp8 A and W loaded via cp.async → lossless fp8→bf16 upcast in shmem → bf16 MMA
//   → fp32 inner partial per kblock → scale by (A_row_scale * W_col_scale) in fp32
//   → fp32 outer accumulator → write fp32 C.
// Precision matches Triton's `tl.dot(a_fp8, w_fp8.T) + acc * scales` path.
__global__ void gemm_fp8_fp8_grouped_kernel(
    const __nv_fp8_e4m3* __restrict__ A_base,     // [N_sel, K] fp8
    const float*         __restrict__ A_scale,    // [N_sel, KB]
    const __nv_fp8_e4m3* __restrict__ W_base,     // [E_LOCAL, N, K] fp8
    const float*         __restrict__ S_base,     // [E_LOCAL, NB, KB]
    const int32_t*       __restrict__ offs,
    const int32_t*       __restrict__ sched_e,
    const int32_t*       __restrict__ sched_tm,
    int N, int K, int NB, int KB,
    float*        __restrict__ C_base,            // [N_sel, ldC] fp32
    int ldC)
{
  const int e    = sched_e[blockIdx.y];
  const int tm   = sched_tm[blockIdx.y];
  const int rs   = offs[e];
  const int re   = offs[e + 1];
  const int M    = re - rs;
  const int m0   = tm * BM;
  const int n0   = blockIdx.x * BN;
  if (m0 >= M) return;

  const __nv_fp8_e4m3* A       = A_base   + (size_t)rs * (size_t)K;
  const float*         Asc     = A_scale  + (size_t)rs * (size_t)KB;
  const __nv_fp8_e4m3* B_fp8_g = W_base   + (size_t)e  * (size_t)N * (size_t)K;
  const float*         B_scale = S_base   + (size_t)e  * (size_t)NB * (size_t)KB;
  float*               C       = C_base   + (size_t)rs * (size_t)ldC;

  const int tid          = threadIdx.x;
  const int warp         = tid >> 5;
  const int lane         = tid & 31;
  const int warp_m       = warp / NWARPS_N_FP8;
  const int warp_n       = warp % NWARPS_N_FP8;
  const int nb           = n0 / BS;
  const int group_id     = lane >> 2;     // 0..7
  const int tid_in_group = lane & 3;      // 0..3

  // Shmem layout (2-stage fp8 pipeline, single-buffer bf16 convert):
  //   sA_fp8 [2][BM][LDA_FP8_PAD]  — 36864 B
  //   sB_fp8 [2][BN][LDB_FP8_PAD]  — 36864
  //   sA_bf  [BM][LDA_BF16_PAD]    — 34816 B
  //   sB_bf  [BN][LDB_BF16_PAD]    — 34816
  //   Total: 143360 B ≈ 140 KB
  extern __shared__ __align__(16) char smem_raw[];
  char* sp = smem_raw;
  __nv_fp8_e4m3 (*sA_fp8)[BM][LDA_FP8_PAD] =
      reinterpret_cast<__nv_fp8_e4m3 (*)[BM][LDA_FP8_PAD]>(sp);
  sp += SA_FP8_STG_BYTES;
  __nv_fp8_e4m3 (*sB_fp8)[BN][LDB_FP8_PAD] =
      reinterpret_cast<__nv_fp8_e4m3 (*)[BN][LDB_FP8_PAD]>(sp);
  sp += SB_FP8_STG_BYTES;
  __nv_bfloat16 (*sA_bf)[LDA_BF16_PAD] =
      reinterpret_cast<__nv_bfloat16 (*)[LDA_BF16_PAD]>(sp);
  sp += SA_BF16_BYTES;
  __nv_bfloat16 (*sB_bf)[LDB_BF16_PAD] =
      reinterpret_cast<__nv_bfloat16 (*)[LDB_BF16_PAD]>(sp);

  // Warp-tile geometry: WM=32, WN=64, 2×8 = 16 m16n8 tiles per warp.
  constexpr int WN_TILES = WN_FP8 / 8;  // 8

  // Outer fp32 accumulator (final result).
  float outer[W_TILES_M_FP8][WN_TILES][4];
  #pragma unroll
  for (int i = 0; i < W_TILES_M_FP8; i++)
    #pragma unroll
    for (int j = 0; j < WN_TILES; j++)
      #pragma unroll
      for (int q = 0; q < 4; q++) outer[i][j][q] = 0.0f;

  // Loads use 16 bytes per thread; BM*BK=16384 bytes → 4 iters @ 256 threads × 16.
  constexpr int LOAD_ITERS = (BM * BK_FP8) / (THREADS_FP8 * 16);
  static_assert((BM * BK_FP8) % (THREADS_FP8 * 16) == 0, "A/B load not tileable");

  // ── Prologue: prefetch stage 0, then (if possible) stage 1. ──
  // 3-stage pipeline: main loop keeps 2 groups in flight while consuming 1.
  auto issue_load_stage = [&](int stage_idx, int kb_idx) {
    int k0 = kb_idx * BK_FP8;
    #pragma unroll
    for (int it = 0; it < LOAD_ITERS; it++) {
      int id = tid + it * THREADS_FP8;
      int r  = id / (BK_FP8 / 16);
      int c  = (id % (BK_FP8 / 16)) * 16;
      bool ok = (m0 + r) < M;
      const void* gptr = &A[(size_t)(m0 + r) * K + (k0 + c)];
      cp_async_16(&sA_fp8[stage_idx][r][c], gptr, ok);
    }
    #pragma unroll
    for (int it = 0; it < LOAD_ITERS; it++) {
      int id = tid + it * THREADS_FP8;
      int r  = id / (BK_FP8 / 16);
      int c  = (id % (BK_FP8 / 16)) * 16;
      bool ok = (n0 + r) < N;
      const void* gptr = &B_fp8_g[(size_t)(n0 + r) * K + (k0 + c)];
      cp_async_16(&sB_fp8[stage_idx][r][c], gptr, ok);
    }
  };

  issue_load_stage(0, 0);
  cp_async_commit();
  if (KB > 1) {
    issue_load_stage(1, 1);
    cp_async_commit();
  }

  int stage = 0;
  for (int kb = 0; kb < KB; kb++) {
    // ── Prefetch kb+2 (if any); wait for stage (kb) load to complete. ──
    int prefetch_kb = kb + 2;
    if (prefetch_kb < KB) {
      int prefetch_stage = prefetch_kb % GEMM1_STAGES;
      issue_load_stage(prefetch_stage, prefetch_kb);
      cp_async_commit();
      cp_async_wait<2>();    // 2 groups still in flight
    } else if (kb + 1 < KB) {
      cp_async_wait<1>();    // 1 group still in flight
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    // ── Lossless convert sA_fp8[stage] → sA_bf (16 fp8 → 16 bf16 per thread per iter). ──
    #pragma unroll
    for (int it = 0; it < LOAD_ITERS; it++) {
      int id = tid + it * THREADS_FP8;
      int r  = id / (BK_FP8 / 16);
      int c  = (id % (BK_FP8 / 16)) * 16;
      uint4 u = *reinterpret_cast<const uint4*>(&sA_fp8[stage][r][c]);
      __nv_fp8_e4m3 bytes[16];
      *reinterpret_cast<uint4*>(bytes) = u;
      __nv_bfloat16 dst[16];
      #pragma unroll
      for (int q = 0; q < 16; q++) dst[q] = __float2bfloat16((float)bytes[q]);
      reinterpret_cast<uint4*>(&sA_bf[r][c])[0] = reinterpret_cast<const uint4*>(dst)[0];
      reinterpret_cast<uint4*>(&sA_bf[r][c])[1] = reinterpret_cast<const uint4*>(dst)[1];
    }
    // ── Lossless convert sB_fp8[stage] → sB_bf. ──
    #pragma unroll
    for (int it = 0; it < LOAD_ITERS; it++) {
      int id = tid + it * THREADS_FP8;
      int r  = id / (BK_FP8 / 16);
      int c  = (id % (BK_FP8 / 16)) * 16;
      uint4 u = *reinterpret_cast<const uint4*>(&sB_fp8[stage][r][c]);
      __nv_fp8_e4m3 bytes[16];
      *reinterpret_cast<uint4*>(bytes) = u;
      __nv_bfloat16 dst[16];
      #pragma unroll
      for (int q = 0; q < 16; q++) dst[q] = __float2bfloat16((float)bytes[q]);
      reinterpret_cast<uint4*>(&sB_bf[r][c])[0] = reinterpret_cast<const uint4*>(dst)[0];
      reinterpret_cast<uint4*>(&sB_bf[r][c])[1] = reinterpret_cast<const uint4*>(dst)[1];
    }
    __syncthreads();

    // ── Inner fp32 accumulator (per-kblock). Zero then accumulate 8 bf16 mmas. ──
    float inner[W_TILES_M_FP8][WN_TILES][4];
    #pragma unroll
    for (int i = 0; i < W_TILES_M_FP8; i++)
      #pragma unroll
      for (int j = 0; j < WN_TILES; j++)
        #pragma unroll
        for (int q = 0; q < 4; q++) inner[i][j][q] = 0.0f;

    // m16n8k16 bf16 MMA: 128 K / 16 = 8 sub-iters.
    #pragma unroll
    for (int kk = 0; kk < BK_FP8; kk += 16) {
      uint32_t Aregs[W_TILES_M_FP8][4];
      #pragma unroll
      for (int i = 0; i < W_TILES_M_FP8; i++) {
        int rbase = warp_m * WM_FP8 + i * 16;
        int row0  = rbase + group_id;
        int row1  = rbase + group_id + 8;
        int col0  = kk + tid_in_group * 2;
        int col1  = col0 + 8;
        Aregs[i][0] = *reinterpret_cast<const uint32_t*>(&sA_bf[row0][col0]);
        Aregs[i][1] = *reinterpret_cast<const uint32_t*>(&sA_bf[row1][col0]);
        Aregs[i][2] = *reinterpret_cast<const uint32_t*>(&sA_bf[row0][col1]);
        Aregs[i][3] = *reinterpret_cast<const uint32_t*>(&sA_bf[row1][col1]);
      }
      uint32_t Bregs[WN_TILES][2];
      #pragma unroll
      for (int j = 0; j < WN_TILES; j++) {
        int cbase = warp_n * WN_FP8 + j * 8;
        int col_n = cbase + group_id;
        int rowk0 = kk + tid_in_group * 2;
        int rowk1 = rowk0 + 8;
        Bregs[j][0] = *reinterpret_cast<const uint32_t*>(&sB_bf[col_n][rowk0]);
        Bregs[j][1] = *reinterpret_cast<const uint32_t*>(&sB_bf[col_n][rowk1]);
      }
      #pragma unroll
      for (int i = 0; i < W_TILES_M_FP8; i++) {
        #pragma unroll
        for (int j = 0; j < WN_TILES; j++) {
          mma_m16n8k16_bf16(
              inner[i][j][0], inner[i][j][1], inner[i][j][2], inner[i][j][3],
              Aregs[i][0], Aregs[i][1], Aregs[i][2], Aregs[i][3],
              Bregs[j][0], Bregs[j][1]);
        }
      }
    }

    // ── Apply per-kblock scales (in fp32), accumulate into outer. ──
    float w_sc = B_scale[(size_t)nb * KB + kb];
    #pragma unroll
    for (int i = 0; i < W_TILES_M_FP8; i++) {
      int rbase    = warp_m * WM_FP8 + i * 16;
      int rlo_glob = m0 + rbase + group_id;
      int rhi_glob = rlo_glob + 8;
      float a_lo = (rlo_glob < M) ? Asc[(size_t)rlo_glob * KB + kb] : 0.0f;
      float a_hi = (rhi_glob < M) ? Asc[(size_t)rhi_glob * KB + kb] : 0.0f;
      float s_lo = a_lo * w_sc;
      float s_hi = a_hi * w_sc;
      #pragma unroll
      for (int j = 0; j < WN_TILES; j++) {
        outer[i][j][0] += inner[i][j][0] * s_lo;
        outer[i][j][1] += inner[i][j][1] * s_lo;
        outer[i][j][2] += inner[i][j][2] * s_hi;
        outer[i][j][3] += inner[i][j][3] * s_hi;
      }
    }

    stage = (stage + 1) % GEMM1_STAGES;
    // Note: no trailing __syncthreads here. The next iter's sync after cp.async
    // wait (or the epilogue on the final iter) serves as the barrier before
    // sA_bf/sB_bf get overwritten.
  }

  // Epilogue: write outer fp32 to C.
  #pragma unroll
  for (int i = 0; i < W_TILES_M_FP8; i++) {
    int rbase = m0 + warp_m * WM_FP8 + i * 16;
    int rlo   = rbase + group_id;
    int rhi   = rlo + 8;
    #pragma unroll
    for (int j = 0; j < WN_TILES; j++) {
      int cbase = n0 + warp_n * WN_FP8 + j * 8;
      int clo   = cbase + tid_in_group * 2;
      int chi   = clo + 1;
      if (rlo < M && clo < N) C[(size_t)rlo * ldC + clo] = outer[i][j][0];
      if (rlo < M && chi < N) C[(size_t)rlo * ldC + chi] = outer[i][j][1];
      if (rhi < M && clo < N) C[(size_t)rhi * ldC + clo] = outer[i][j][2];
      if (rhi < M && chi < N) C[(size_t)rhi * ldC + chi] = outer[i][j][3];
    }
  }
}

// ======================================================================
// Warp-specialized FP8 GEMM1 — EXPERIMENTAL, CURRENTLY DISABLED.
//
// Design:
// - Tile: BM=128, BN=64, BK_FP8=128 (one kblock per main-loop iter).
// - 8 warps per block: warps 0-3 = producer (128 threads, all cp.async
//   for A and B), warps 4-7 = consumer (128 threads, all MMA).
// - 3-stage shmem pipeline synchronized with mbarriers.
//   * mbar_ready[s]: producer → consumer signal; init count = 128 (every
//     producer thread arrives per stage).
//   * mbar_consumed[s]: consumer → producer signal; init count = 128.
// - __launch_bounds__(256, 2) targets 2 blocks/SM (→ 25% occupancy).
//
// Status (2026-04-23):
// 1. With a synchronous `cp_async_wait<0>` + explicit `mbarrier.arrive`
//    in the producer, correctness passes (match_ratio 0.998 on wl0) but
//    it's ~45% slower than the non-WSP kernel (9.04ms vs 6.23ms on wl8).
//    Root cause: cp_async_wait<0> serializes each stage's loads within
//    the producer, so there's never more than one stage in flight —
//    defeating the whole point of warp specialization. And we've also
//    halved the compute parallelism (4 consumer warps vs 8 all-compute).
// 2. Switching to `cp.async.mbarrier.arrive.b64` (async attach — correct
//    form per PTX ISA 8.x, no state-space qualifier) hangs. Either the
//    barrier-arrival isn't firing as I expect or there's a subtle
//    ordering bug with `cp_async_commit_group`. The PTX assembled and
//    the block eventually deadlocks with the consumer still waiting on
//    mbar_ready[0].
//
// To finish this, the right move is to switch the gmem→shmem path to
// TMA (cp.async.bulk.tensor.2d) — the bulk form has a first-class
// mbarrier argument, bypassing the commit_group dance entirely and is
// the Hopper+/Blackwell idiom. Kernel left in the file so a future
// attempt can iterate on it.
// ======================================================================
constexpr int BN_WSP             = 128;
constexpr int BK_WSP             = BK_FP8;   // 128
constexpr int LDA_WSP_PAD        = BK_WSP;
constexpr int LDB_WSP_PAD        = BK_WSP + 16;
constexpr int STAGES_WSP         = 2;
constexpr int NPROD_WARPS        = 4;
constexpr int NCONS_WARPS        = 4;
constexpr int NWARPS_WSP         = NPROD_WARPS + NCONS_WARPS;          // 8
constexpr int THREADS_WSP        = NWARPS_WSP * 32;                     // 256
constexpr int NPROD_THREADS      = NPROD_WARPS * 32;                    // 128
constexpr int NCONS_THREADS      = NCONS_WARPS * 32;                    // 128
constexpr int NWARPS_M_CONS      = 4;
constexpr int NWARPS_N_CONS      = 2;
constexpr int WM_CONS            = BM / NWARPS_M_CONS;                  // 32
constexpr int WN_CONS            = BN_WSP / NWARPS_N_CONS;              // 64
constexpr int W_TILES_M_CONS     = WM_CONS / 16;                        // 4
constexpr int W_TILES_N_CONS     = WN_CONS / 8;                         // 4
constexpr size_t SA_WSP_BYTES    = (size_t)STAGES_WSP * BM     * LDA_WSP_PAD;
constexpr size_t SB_WSP_BYTES    = (size_t)STAGES_WSP * BN_WSP * LDB_WSP_PAD;
constexpr size_t GEMM_WSP_SMEM_BYTES =
    SA_WSP_BYTES + SB_WSP_BYTES + 128;  // dynamic base alignment slack

template <int THREADS>
__device__ __forceinline__ void stage_load_A_wsp(
    __nv_fp8_e4m3 (*sA)[LDA_WSP_PAD],
    const __nv_fp8_e4m3* A, int m0, int k0, int M, int K, int tid)
{
  constexpr int ITERS = (BM * BK_WSP) / (THREADS * 16);
  static_assert((BM * BK_WSP) % (THREADS * 16) == 0, "A load not tileable");
  #pragma unroll
  for (int it = 0; it < ITERS; it++) {
    int id = tid + it * THREADS;
    int r  = id / (BK_WSP / 16);
    int c  = (id % (BK_WSP / 16)) * 16;
    bool ok = (m0 + r) < M;
    const void* gptr = &A[(size_t)(m0 + r) * K + (k0 + c)];
    cp_async_16(&sA[r][c], gptr, ok);
  }
}

template <int THREADS>
__device__ __forceinline__ void stage_load_B_wsp(
    __nv_fp8_e4m3 (*sB)[LDB_WSP_PAD],
    const __nv_fp8_e4m3* B, int n0, int k0, int N, int K, int tid)
{
  constexpr int ITERS = (BN_WSP * BK_WSP) / (THREADS * 16);
  static_assert((BN_WSP * BK_WSP) % (THREADS * 16) == 0, "B load not tileable");
  #pragma unroll
  for (int it = 0; it < ITERS; it++) {
    int id = tid + it * THREADS;
    int r  = id / (BK_WSP / 16);
    int c  = (id % (BK_WSP / 16)) * 16;
    bool ok = (n0 + r) < N;
    const void* gptr = &B[(size_t)(n0 + r) * K + (k0 + c)];
    cp_async_16(&sB[r][c], gptr, ok);
  }
}

__global__ __launch_bounds__(THREADS_WSP, 2)
void gemm_fp8_fp8_wsp_grouped_kernel(
    const __nv_fp8_e4m3* __restrict__ A_base,     // [N_sel, K] fp8
    const float*         __restrict__ A_scale,    // [N_sel, KB]
    const __nv_fp8_e4m3* __restrict__ W_base,     // [E, N, K] fp8
    const float*         __restrict__ S_base,     // [E, NB, KB]
    const __grid_constant__ CUtensorMap A_tma,
    const __grid_constant__ CUtensorMap W_tma,
    const int32_t*       __restrict__ offs,
    const int32_t*       __restrict__ sched_e,
    const int32_t*       __restrict__ sched_tm,
    int N, int K, int NB, int KB,
    float*        __restrict__ C_base,            // [N_sel, ldC] fp32
    int ldC)
{
  const int e    = sched_e[blockIdx.y];
  const int tm   = sched_tm[blockIdx.y];
  const int rs   = offs[e];
  const int re   = offs[e + 1];
  const int M    = re - rs;
  const int m0   = tm * BM;
  const int n0   = blockIdx.x * BN_WSP;
  if (m0 >= M) return;

  const float*         Asc     = A_scale  + (size_t)rs * (size_t)KB;
  const __nv_fp8_e4m3* B_fp8   = W_base   + (size_t)e  * (size_t)N * (size_t)K;
  const float*         B_scale = S_base   + (size_t)e  * (size_t)NB * (size_t)KB;
  float*               C       = C_base   + (size_t)rs * (size_t)ldC;

  const int tid          = threadIdx.x;
  const int warp         = tid >> 5;
  const int lane         = tid & 31;
  const int nb           = n0 / BS;
  const int group_id     = lane >> 2;
  const int tid_in_group = lane & 3;

  // Shared memory layout:
  //   [0                 ..)      sA[STAGES][BM][LDA_WSP_PAD]
  //   [SA_BYTES          ..)      sB[STAGES][BN_WSP][LDB_WSP_PAD]
  extern __shared__ __align__(128) char smem_raw[];
  char* smem_base = reinterpret_cast<char*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 127ULL) & ~127ULL);
  __nv_fp8_e4m3 (*sA)[BM][LDA_WSP_PAD] =
      reinterpret_cast<__nv_fp8_e4m3 (*)[BM][LDA_WSP_PAD]>(smem_base);
  __nv_fp8_e4m3 (*sB)[BN_WSP][LDB_WSP_PAD] =
      reinterpret_cast<__nv_fp8_e4m3 (*)[BN_WSP][LDB_WSP_PAD]>(smem_base + SA_WSP_BYTES);
  __shared__ __align__(8) uint64_t mbar_ready[STAGES_WSP];

  // Ready barriers track the combined A+B bulk-copy bytes for each stage.
  if (tid < STAGES_WSP) {
    mbarrier_init_shared(&mbar_ready[tid],  1);
  }
  __syncthreads();
  if (tid == 0) {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("prefetch.tensormap [%0];\n"
                 :: "l"(reinterpret_cast<uint64_t>(&A_tma)) : "memory");
    asm volatile("prefetch.tensormap [%0];\n"
                 :: "l"(reinterpret_cast<uint64_t>(&W_tma)) : "memory");
  }
  __syncthreads();

  if (warp == 0 && lane == 0) {
    constexpr uint32_t A_BYTES = BM * BK_WSP;
    mbarrier_arrive_expect_tx_shared(&mbar_ready[0], A_BYTES);
    cp_async_bulk_tensor_2d_shared_global(&sA[0][0][0],
                                          &A_tma, 0, rs + m0,
                                          &mbar_ready[0]);
  }
  stage_load_B_wsp<THREADS_WSP>(sB[0], B_fp8, n0, 0, N, K, tid);
  cp_async_commit();

  // All eight warps compute the original 4x2 warp tile after one elected
  // lane issues the next TMA copy.
  const int cons_warp = warp;
  const int warp_m    = cons_warp / NWARPS_N_CONS;
  const int warp_n    = cons_warp % NWARPS_N_CONS;

  float outer[W_TILES_M_CONS][W_TILES_N_CONS][4];
  #pragma unroll
  for (int i = 0; i < W_TILES_M_CONS; i++)
    #pragma unroll
    for (int j = 0; j < W_TILES_N_CONS; j++)
      #pragma unroll
      for (int q = 0; q < 4; q++) outer[i][j][q] = 0.0f;

  for (int kb = 0; kb < KB; kb++) {
    int stage = kb & 1;
    uint32_t rphase = (kb >> 1) & 1;

    int next_kb = kb + 1;
    if (next_kb < KB) {
      int next_stage = next_kb & 1;
      int next_k_off = next_kb * BK_WSP;
      if (warp == 0 && lane == 0) {
        constexpr uint32_t A_BYTES = BM * BK_WSP;
        mbarrier_arrive_expect_tx_shared(&mbar_ready[next_stage], A_BYTES);
        cp_async_bulk_tensor_2d_shared_global(&sA[next_stage][0][0],
                                              &A_tma, next_k_off, rs + m0,
                                              &mbar_ready[next_stage]);
      }
      stage_load_B_wsp<THREADS_WSP>(sB[next_stage], B_fp8, n0, next_k_off, N, K, tid);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    mbarrier_wait_parity_shared(&mbar_ready[stage], rphase);
    __syncthreads();

    float w_sc = B_scale[(size_t)nb * KB + kb];
    float scale_lo[W_TILES_M_CONS];
    float scale_hi[W_TILES_M_CONS];
    #pragma unroll
    for (int i = 0; i < W_TILES_M_CONS; i++) {
      int rbase    = warp_m * WM_CONS + i * 16;
      int rlo_glob = m0 + rbase + group_id;
      int rhi_glob = rlo_glob + 8;
      float a_lo = (rlo_glob < M) ? Asc[(size_t)rlo_glob * KB + kb] : 0.0f;
      float a_hi = (rhi_glob < M) ? Asc[(size_t)rhi_glob * KB + kb] : 0.0f;
      scale_lo[i] = a_lo * w_sc;
      scale_hi[i] = a_hi * w_sc;
    }

    #pragma unroll
    for (int kk = 0; kk < BK_WSP; kk += 32) {
      uint32_t Aregs[W_TILES_M_CONS][4];
      #pragma unroll
      for (int i = 0; i < W_TILES_M_CONS; i++) {
        int rbase = warp_m * WM_CONS + i * 16;
        int row0  = rbase + group_id;
        int row1  = rbase + group_id + 8;
        int col0  = kk + tid_in_group * 4;
        int col1  = col0 + 16;
        Aregs[i][0] = *reinterpret_cast<const uint32_t*>(&sA[stage][row0][col0]);
        Aregs[i][1] = *reinterpret_cast<const uint32_t*>(&sA[stage][row1][col0]);
        Aregs[i][2] = *reinterpret_cast<const uint32_t*>(&sA[stage][row0][col1]);
        Aregs[i][3] = *reinterpret_cast<const uint32_t*>(&sA[stage][row1][col1]);
      }
      uint32_t Bregs[W_TILES_N_CONS][2];
      #pragma unroll
      for (int j = 0; j < W_TILES_N_CONS; j++) {
        int cbase = warp_n * WN_CONS + j * 8;
        int col_n = cbase + group_id;
        int rowk0 = kk + tid_in_group * 4;
        int rowk1 = rowk0 + 16;
        Bregs[j][0] = *reinterpret_cast<const uint32_t*>(&sB[stage][col_n][rowk0]);
        Bregs[j][1] = *reinterpret_cast<const uint32_t*>(&sB[stage][col_n][rowk1]);
      }
      #pragma unroll
      for (int i = 0; i < W_TILES_M_CONS; i++) {
        #pragma unroll
        for (int j = 0; j < W_TILES_N_CONS; j++) {
          float acc0 = 0.0f;
          float acc1 = 0.0f;
          float acc2 = 0.0f;
          float acc3 = 0.0f;
          mma_m16n8k32_e4m3(
              acc0, acc1, acc2, acc3,
              Aregs[i][0], Aregs[i][1], Aregs[i][2], Aregs[i][3],
              Bregs[j][0], Bregs[j][1]);
          outer[i][j][0] += acc0 * scale_lo[i];
          outer[i][j][1] += acc1 * scale_lo[i];
          outer[i][j][2] += acc2 * scale_hi[i];
          outer[i][j][3] += acc3 * scale_hi[i];
        }
      }
    }
    __syncthreads();
  }

  // Epilogue: write outer fp32 to C.
  #pragma unroll
  for (int i = 0; i < W_TILES_M_CONS; i++) {
    int rbase = m0 + warp_m * WM_CONS + i * 16;
    int rlo   = rbase + group_id;
    int rhi   = rlo + 8;
    #pragma unroll
    for (int j = 0; j < W_TILES_N_CONS; j++) {
      int cbase = n0 + warp_n * WN_CONS + j * 8;
      int clo   = cbase + tid_in_group * 2;
      int chi   = clo + 1;
      if (rlo < M && clo < N) C[(size_t)rlo * ldC + clo] = outer[i][j][0];
      if (rlo < M && chi < N) C[(size_t)rlo * ldC + chi] = outer[i][j][1];
      if (rhi < M && clo < N) C[(size_t)rhi * ldC + clo] = outer[i][j][2];
      if (rhi < M && chi < N) C[(size_t)rhi * ldC + chi] = outer[i][j][3];
    }
  }
}

// Vectorized gather: one warp per row handles 32×16 = 512 fp8 bytes per step;
// grid covers H = NH_BLKS*BS = 7168 with ceil(7168/512) = 14 waves per warp.
// Fewer, fatter blocks than the old (N_sel, 56)×128 layout cut the launch
// count ~14× for typical workloads.
__global__ void gather_hs_fp8_kernel(
    const __nv_fp8_e4m3* __restrict__ hs_fp8,
    const float*         __restrict__ hs_scale,
    const int32_t*       __restrict__ perm_token,
    int T, int N_sel,
    __nv_fp8_e4m3* __restrict__ A_fp8,
    float* __restrict__ A_scale)
{
  const int r    = blockIdx.x;
  if (r >= N_sel) return;
  const int t    = perm_token[r];
  const int tid  = threadIdx.x;              // 0..127
  if (blockIdx.y == 0 && tid < NH_BLKS) {
    A_scale[(size_t)r * NH_BLKS + tid] = hs_scale[(size_t)tid * T + t];
  }
  // Each thread moves 16 fp8 bytes per wave; grid.y tiles H in chunks.
  const int base = blockIdx.y * (blockDim.x * 16) + tid * 16;
  if (base >= H) return;
  *reinterpret_cast<uint4*>(&A_fp8[(size_t)r * H + base]) =
      *reinterpret_cast<const uint4*>(&hs_fp8[(size_t)t * H + base]);
}

// ======================================================================
// Kernel 9: SwiGLU.  G1 [M, 2I] -> out [M, I]; out = silu(G1[:, I:]) * G1[:, :I]
// ======================================================================
// Vectorized swiglu: 4 elements per thread. fp32 in, fp32 out.
// Keeping C in fp32 preserves precision for GEMM2 (consumed via bf16x2 emulation).
__global__ void swiglu_kernel(
    const float* __restrict__ G1,
    int M,
    float* __restrict__ out)
{
  const int r  = blockIdx.y;
  const int d4 = blockIdx.x * blockDim.x + threadIdx.x;
  const int d  = d4 * 4;
  if (r >= M || d >= I) return;
  float4 x1 = *reinterpret_cast<const float4*>(&G1[(size_t)r * II + d]);
  float4 x2 = *reinterpret_cast<const float4*>(&G1[(size_t)r * II + I + d]);
  float x1a[4] = {x1.x, x1.y, x1.z, x1.w};
  float x2a[4] = {x2.x, x2.y, x2.z, x2.w};
  float4 o;
  {
    float silu = x2a[0] / (1.0f + __expf(-x2a[0])); o.x = silu * x1a[0];
  }
  {
    float silu = x2a[1] / (1.0f + __expf(-x2a[1])); o.y = silu * x1a[1];
  }
  {
    float silu = x2a[2] / (1.0f + __expf(-x2a[2])); o.z = silu * x1a[2];
  }
  {
    float silu = x2a[3] / (1.0f + __expf(-x2a[3])); o.w = silu * x1a[3];
  }
  *reinterpret_cast<float4*>(&out[(size_t)r * I + d]) = o;
}

// ======================================================================
// GEMM2 with bf16x2 emulation for fp32 A × fp8 W (+ per-block scale).
// Precision: fp32 A is split into (a_hi=bf16(a), a_lo=bf16(a-bf16(a))) giving
// ~14-bit effective mantissa, enough that accumulated error over K=2048 is
// ~0.45% RSS (well under the 1% rtol). Fused per-token weighted atomic scatter.
// ======================================================================
constexpr int LDA_FP32_PAD     = BK_FP8 + 4;   // 132 fp32 cols/row (512B+pad)
constexpr int LDB_FP8_G2_PAD   = BK_FP8 + 16;  // 144 fp8  cols/row
constexpr int LDB_BF16_G2_PAD  = BK_FP8 + 8;   // 136 bf16 cols/row
constexpr int GEMM2_STAGES     = 2;
// fp32 A + fp8 W loads are 2-stage (overlap next fp32 load + fp8 load with
// current bf16x2 mma); bf16 W is single-buffer (consumed within the iter).
constexpr size_t SA_FP32_G2_BYTES  = (size_t)GEMM2_STAGES * BM * LDA_FP32_PAD * sizeof(float);   // 135168
constexpr size_t SB_FP8_G2_BYTES   = (size_t)GEMM2_STAGES * BN * LDB_FP8_G2_PAD;                 // 36864
constexpr size_t SB_BF16_G2_BYTES  = (size_t)BN * LDB_BF16_G2_PAD * sizeof(__nv_bfloat16);        // 34816
constexpr size_t GEMM_G2_SMEM_BYTES = SA_FP32_G2_BYTES + SB_FP8_G2_BYTES + SB_BF16_G2_BYTES;     // ~202 KB

__global__ void gemm_fp32_fp8w_fused_acc_grouped_kernel(
    const float*         __restrict__ A_base,     // [N_sel, K=I] fp32
    const __nv_fp8_e4m3* __restrict__ W_base,     // [E, N=H, K=I] fp8
    const float*         __restrict__ S_base,     // [E, NB=H/128, KB=I/128]
    const int32_t*       __restrict__ offs,
    const int32_t*       __restrict__ sched_e,
    const int32_t*       __restrict__ sched_tm,
    const int32_t*       __restrict__ perm_token,
    const float*         __restrict__ perm_weight,
    int N, int K, int NB, int KB,
    float* __restrict__ acc_out,
    int ldAcc)
{
  const int e    = sched_e[blockIdx.y];
  const int tm   = sched_tm[blockIdx.y];
  const int rs   = offs[e];
  const int re   = offs[e + 1];
  const int M    = re - rs;
  const int m0   = tm * BM;
  const int n0   = blockIdx.x * BN;
  if (m0 >= M) return;

  const float*         A       = A_base  + (size_t)rs * (size_t)K;
  const __nv_fp8_e4m3* B_fp8_g = W_base  + (size_t)e  * (size_t)N * (size_t)K;
  const float*         B_scale = S_base  + (size_t)e  * (size_t)NB * (size_t)KB;
  const int32_t*       Ptok    = perm_token + rs;
  const float*         Pwgt    = perm_weight + rs;

  const int tid          = threadIdx.x;
  const int warp         = tid >> 5;
  const int lane         = tid & 31;
  const int warp_m       = warp / NWARPS_N_FP8;
  const int warp_n       = warp % NWARPS_N_FP8;
  const int nb           = n0 / BS;
  const int group_id     = lane >> 2;
  const int tid_in_group = lane & 3;

  extern __shared__ __align__(16) char smem_raw[];
  char* sp = smem_raw;
  float (*sA_fp32)[BM][LDA_FP32_PAD] =
      reinterpret_cast<float (*)[BM][LDA_FP32_PAD]>(sp);
  sp += SA_FP32_G2_BYTES;
  __nv_fp8_e4m3 (*sB_fp8)[BN][LDB_FP8_G2_PAD] =
      reinterpret_cast<__nv_fp8_e4m3 (*)[BN][LDB_FP8_G2_PAD]>(sp);
  sp += SB_FP8_G2_BYTES;
  __nv_bfloat16 (*sB_bf)[LDB_BF16_G2_PAD] =
      reinterpret_cast<__nv_bfloat16 (*)[LDB_BF16_G2_PAD]>(sp);

  constexpr int WN_TILES = WN_FP8 / 8;  // 8

  float outer[W_TILES_M_FP8][WN_TILES][4];
  #pragma unroll
  for (int i = 0; i < W_TILES_M_FP8; i++)
    #pragma unroll
    for (int j = 0; j < WN_TILES; j++)
      #pragma unroll
      for (int q = 0; q < 4; q++) outer[i][j][q] = 0.0f;

  // A load: fp32, 4 elems/thread/iter = 16 bytes. BM*BK*4=65536 bytes → 16 iters.
  constexpr int A_ITERS = (BM * BK_FP8 * 4) / (THREADS_FP8 * 16);
  static_assert((BM * BK_FP8 * 4) % (THREADS_FP8 * 16) == 0, "A fp32 load not tileable");
  // B load: fp8, 16 fp8/thread/iter. BM*BK=16384 bytes → 4 iters.
  constexpr int B_ITERS = (BN * BK_FP8) / (THREADS_FP8 * 16);
  static_assert((BN * BK_FP8) % (THREADS_FP8 * 16) == 0, "B fp8 load not tileable");

  // ── Prologue: preload stage 0 (fp32 A and fp8 W). ──
  {
    const int k0 = 0;
    #pragma unroll
    for (int it = 0; it < A_ITERS; it++) {
      int id = tid + it * THREADS_FP8;
      int r  = id / (BK_FP8 / 4);
      int c  = (id % (BK_FP8 / 4)) * 4;
      bool ok = (m0 + r) < M;
      const void* gptr = &A[(size_t)(m0 + r) * K + (k0 + c)];
      cp_async_16(&sA_fp32[0][r][c], gptr, ok);
    }
    #pragma unroll
    for (int it = 0; it < B_ITERS; it++) {
      int id = tid + it * THREADS_FP8;
      int r  = id / (BK_FP8 / 16);
      int c  = (id % (BK_FP8 / 16)) * 16;
      bool ok = (n0 + r) < N;
      const void* gptr = &B_fp8_g[(size_t)(n0 + r) * K + (k0 + c)];
      cp_async_16(&sB_fp8[0][r][c], gptr, ok);
    }
    cp_async_commit();
  }

  int stage = 0;
  for (int kb = 0; kb < KB; kb++) {
    // ── Prefetch next stage (if any); wait for current stage load. ──
    int next_kb = kb + 1;
    if (next_kb < KB) {
      int next_stage = 1 - stage;
      int next_k0    = next_kb * BK_FP8;
      #pragma unroll
      for (int it = 0; it < A_ITERS; it++) {
        int id = tid + it * THREADS_FP8;
        int r  = id / (BK_FP8 / 4);
        int c  = (id % (BK_FP8 / 4)) * 4;
        bool ok = (m0 + r) < M;
        const void* gptr = &A[(size_t)(m0 + r) * K + (next_k0 + c)];
        cp_async_16(&sA_fp32[next_stage][r][c], gptr, ok);
      }
      #pragma unroll
      for (int it = 0; it < B_ITERS; it++) {
        int id = tid + it * THREADS_FP8;
        int r  = id / (BK_FP8 / 16);
        int c  = (id % (BK_FP8 / 16)) * 16;
        bool ok = (n0 + r) < N;
        const void* gptr = &B_fp8_g[(size_t)(n0 + r) * K + (next_k0 + c)];
        cp_async_16(&sB_fp8[next_stage][r][c], gptr, ok);
      }
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    // Convert fp8 B[stage] → bf16 B (lossless, single buffer).
    #pragma unroll
    for (int it = 0; it < B_ITERS; it++) {
      int id = tid + it * THREADS_FP8;
      int r  = id / (BK_FP8 / 16);
      int c  = (id % (BK_FP8 / 16)) * 16;
      uint4 u = *reinterpret_cast<const uint4*>(&sB_fp8[stage][r][c]);
      __nv_fp8_e4m3 bytes[16];
      *reinterpret_cast<uint4*>(bytes) = u;
      __nv_bfloat16 dst[16];
      #pragma unroll
      for (int q = 0; q < 16; q++) dst[q] = __float2bfloat16((float)bytes[q]);
      reinterpret_cast<uint4*>(&sB_bf[r][c])[0] = reinterpret_cast<const uint4*>(dst)[0];
      reinterpret_cast<uint4*>(&sB_bf[r][c])[1] = reinterpret_cast<const uint4*>(dst)[1];
    }
    __syncthreads();

    // Inner fp32 accumulator (per-kblock).
    float inner[W_TILES_M_FP8][WN_TILES][4];
    #pragma unroll
    for (int i = 0; i < W_TILES_M_FP8; i++)
      #pragma unroll
      for (int j = 0; j < WN_TILES; j++)
        #pragma unroll
        for (int q = 0; q < 4; q++) inner[i][j][q] = 0.0f;

    // 8 sub-iters (bf16 m16n8k16 × 128 = 8 × 16 K elements).
    #pragma unroll
    for (int kk = 0; kk < BK_FP8; kk += 16) {
      // Load 8 fp32 A values per thread per m-tile, split into (hi, lo) bf16.
      uint32_t Aregs_hi[W_TILES_M_FP8][4];
      uint32_t Aregs_lo[W_TILES_M_FP8][4];
      #pragma unroll
      for (int i = 0; i < W_TILES_M_FP8; i++) {
        int rbase = warp_m * WM_FP8 + i * 16;
        int row0  = rbase + group_id;
        int row1  = rbase + group_id + 8;
        int col0  = kk + tid_in_group * 2;
        int col1  = col0 + 8;
        float a0 = sA_fp32[stage][row0][col0    ];
        float a1 = sA_fp32[stage][row0][col0 + 1];
        float a2 = sA_fp32[stage][row1][col0    ];
        float a3 = sA_fp32[stage][row1][col0 + 1];
        float a4 = sA_fp32[stage][row0][col1    ];
        float a5 = sA_fp32[stage][row0][col1 + 1];
        float a6 = sA_fp32[stage][row1][col1    ];
        float a7 = sA_fp32[stage][row1][col1 + 1];

        __nv_bfloat16 h0 = __float2bfloat16(a0), l0 = __float2bfloat16(a0 - __bfloat162float(h0));
        __nv_bfloat16 h1 = __float2bfloat16(a1), l1 = __float2bfloat16(a1 - __bfloat162float(h1));
        __nv_bfloat16 h2 = __float2bfloat16(a2), l2 = __float2bfloat16(a2 - __bfloat162float(h2));
        __nv_bfloat16 h3 = __float2bfloat16(a3), l3 = __float2bfloat16(a3 - __bfloat162float(h3));
        __nv_bfloat16 h4 = __float2bfloat16(a4), l4 = __float2bfloat16(a4 - __bfloat162float(h4));
        __nv_bfloat16 h5 = __float2bfloat16(a5), l5 = __float2bfloat16(a5 - __bfloat162float(h5));
        __nv_bfloat16 h6 = __float2bfloat16(a6), l6 = __float2bfloat16(a6 - __bfloat162float(h6));
        __nv_bfloat16 h7 = __float2bfloat16(a7), l7 = __float2bfloat16(a7 - __bfloat162float(h7));

        // Pack (x, y) as u32 with x in low 16, y in high 16.
        auto pack2 = [](__nv_bfloat16 lo, __nv_bfloat16 hi) -> uint32_t {
          uint16_t lo_u = *reinterpret_cast<uint16_t*>(&lo);
          uint16_t hi_u = *reinterpret_cast<uint16_t*>(&hi);
          return ((uint32_t)hi_u << 16) | (uint32_t)lo_u;
        };
        Aregs_hi[i][0] = pack2(h0, h1);
        Aregs_hi[i][1] = pack2(h2, h3);
        Aregs_hi[i][2] = pack2(h4, h5);
        Aregs_hi[i][3] = pack2(h6, h7);
        Aregs_lo[i][0] = pack2(l0, l1);
        Aregs_lo[i][1] = pack2(l2, l3);
        Aregs_lo[i][2] = pack2(l4, l5);
        Aregs_lo[i][3] = pack2(l6, l7);
      }
      // Load B fragments (bf16).
      uint32_t Bregs[WN_TILES][2];
      #pragma unroll
      for (int j = 0; j < WN_TILES; j++) {
        int cbase = warp_n * WN_FP8 + j * 8;
        int col_n = cbase + group_id;
        int rowk0 = kk + tid_in_group * 2;
        int rowk1 = rowk0 + 8;
        Bregs[j][0] = *reinterpret_cast<const uint32_t*>(&sB_bf[col_n][rowk0]);
        Bregs[j][1] = *reinterpret_cast<const uint32_t*>(&sB_bf[col_n][rowk1]);
      }
      // MMA: inner += (a_hi + a_lo) × B = mma(a_hi, B) + mma(a_lo, B).
      #pragma unroll
      for (int i = 0; i < W_TILES_M_FP8; i++) {
        #pragma unroll
        for (int j = 0; j < WN_TILES; j++) {
          mma_m16n8k16_bf16(
              inner[i][j][0], inner[i][j][1], inner[i][j][2], inner[i][j][3],
              Aregs_hi[i][0], Aregs_hi[i][1], Aregs_hi[i][2], Aregs_hi[i][3],
              Bregs[j][0], Bregs[j][1]);
          mma_m16n8k16_bf16(
              inner[i][j][0], inner[i][j][1], inner[i][j][2], inner[i][j][3],
              Aregs_lo[i][0], Aregs_lo[i][1], Aregs_lo[i][2], Aregs_lo[i][3],
              Bregs[j][0], Bregs[j][1]);
        }
      }
    }

    // Apply W scale (scalar per (nb, kb)), add to outer.
    float w_sc = B_scale[(size_t)nb * KB + kb];
    #pragma unroll
    for (int i = 0; i < W_TILES_M_FP8; i++) {
      #pragma unroll
      for (int j = 0; j < WN_TILES; j++) {
        outer[i][j][0] += inner[i][j][0] * w_sc;
        outer[i][j][1] += inner[i][j][1] * w_sc;
        outer[i][j][2] += inner[i][j][2] * w_sc;
        outer[i][j][3] += inner[i][j][3] * w_sc;
      }
    }

    stage = 1 - stage;
    // Note: no trailing __syncthreads here. Next iter's sync after cp.async
    // wait serves as the barrier before sB_bf is overwritten.
  }

  // Epilogue: fused per-token weighted atomic scatter to fp32 output.
  #pragma unroll
  for (int i = 0; i < W_TILES_M_FP8; i++) {
    int rbase_loc = warp_m * WM_FP8 + i * 16;
    int rlo_loc   = rbase_loc + group_id;
    int rhi_loc   = rlo_loc + 8;
    int gm_lo     = m0 + rlo_loc;
    int gm_hi     = m0 + rhi_loc;
    int t_lo = (gm_lo < M) ? Ptok[gm_lo] : 0;
    int t_hi = (gm_hi < M) ? Ptok[gm_hi] : 0;
    float w_lo = (gm_lo < M) ? Pwgt[gm_lo] : 0.0f;
    float w_hi = (gm_hi < M) ? Pwgt[gm_hi] : 0.0f;
    #pragma unroll
    for (int j = 0; j < WN_TILES; j++) {
      int cbase = n0 + warp_n * WN_FP8 + j * 8;
      int clo   = cbase + tid_in_group * 2;
      int chi   = clo + 1;
      if (gm_lo < M && clo < N)
        atomicAdd(&acc_out[(size_t)t_lo * ldAcc + clo], outer[i][j][0] * w_lo);
      if (gm_lo < M && chi < N)
        atomicAdd(&acc_out[(size_t)t_lo * ldAcc + chi], outer[i][j][1] * w_lo);
      if (gm_hi < M && clo < N)
        atomicAdd(&acc_out[(size_t)t_hi * ldAcc + clo], outer[i][j][2] * w_hi);
      if (gm_hi < M && chi < N)
        atomicAdd(&acc_out[(size_t)t_hi * ldAcc + chi], outer[i][j][3] * w_hi);
    }
  }
}

// ======================================================================
// Kernel 10+11: fp32-accumulate scatter, then cast to bf16.
// ======================================================================
__global__ void fin_acc_kernel(
    const __nv_bfloat16* __restrict__ O,        // [N_sel, H]
    const int32_t* __restrict__ perm_token,     // [N_sel]
    const float*   __restrict__ perm_weight,    // [N_sel]
    int N_sel,
    float* __restrict__ acc)                     // [T, H] fp32
{
  const int r = blockIdx.y;
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= N_sel || d >= H) return;
  int   t = perm_token[r];
  float w = perm_weight[r];
  float v = __bfloat162float(O[(size_t)r * H + d]) * w;
  atomicAdd(&acc[(size_t)t * H + d], v);
}

// Vectorized fin_cast: each thread processes 4 fp32 → 4 bf16 in a uint4/uint2
// transaction per side. Launch with H*T/4 threads to keep 16B granularity.
__global__ void fin_cast_kernel(
    const float* __restrict__ acc,
    int T,
    __nv_bfloat16* __restrict__ output)
{
  const int t = blockIdx.y;
  const int d4 = blockIdx.x * blockDim.x + threadIdx.x;
  const int d  = d4 * 4;
  if (t >= T || d >= H) return;
  const float4 v = *reinterpret_cast<const float4*>(&acc[(size_t)t * H + d]);
  __nv_bfloat16 o[4] = {
      __float2bfloat16(v.x), __float2bfloat16(v.y),
      __float2bfloat16(v.z), __float2bfloat16(v.w)
  };
  *reinterpret_cast<uint2*>(&output[(size_t)t * H + d]) =
      *reinterpret_cast<const uint2*>(o);
}

__global__ void zero_f32_kernel(float* p, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 0.0f;
}

__global__ void zero_i32_kernel(int32_t* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 0;
}

// ======================================================================
// Persistent workspace.
// ======================================================================
struct Workspace {
  int32_t* d_topk_idx    = nullptr;
  float*   d_topk_w      = nullptr;
  int32_t* d_counts      = nullptr;
  int32_t* d_offs        = nullptr;
  int32_t* d_write_ptr   = nullptr;
  int32_t* d_n_sel       = nullptr;
  int32_t* d_perm_token  = nullptr;
  float*   d_perm_weight = nullptr;
  int32_t* d_sched_e     = nullptr;  // grouped-GEMM schedule: expert per M-tile
  int32_t* d_sched_tm    = nullptr;  // grouped-GEMM schedule: tile_m per M-tile
  int32_t* h_sched_e     = nullptr;  // pinned host staging
  int32_t* h_sched_tm    = nullptr;  // pinned host staging
  int      sched_cap     = 0;

  __nv_fp8_e4m3* d_A_fp8      = nullptr;  // fp8 permuted (GEMM1 input path)
  float*         d_A_scale    = nullptr;  // [N_sel, NH_BLKS] per-row-per-kblock
  float*         d_G1         = nullptr;  // fp32 [N_sel, II] (GEMM1 → SwiGLU)
  float*         d_C_fp32     = nullptr;  // fp32 SwiGLU output, GEMM2 input
  float*         d_out_f32    = nullptr;

  int  cap_T          = 0;
  bool singletons_ok  = false;
  bool weights_alloc  = false;
};
static Workspace g_ws;

static void ensure_singletons() {
  if (g_ws.singletons_ok) return;
  cudaMalloc(&g_ws.d_counts,    E_LOCAL       * sizeof(int32_t));
  cudaMalloc(&g_ws.d_offs,      (E_LOCAL + 1) * sizeof(int32_t));
  cudaMalloc(&g_ws.d_write_ptr, E_LOCAL       * sizeof(int32_t));
  cudaMalloc(&g_ws.d_n_sel,     1             * sizeof(int32_t));
  g_ws.singletons_ok = true;
}

static void ensure_schedule(int T) {
  int max_tiles = E_LOCAL + (T * TOP_K + BM - 1) / BM;
  if (g_ws.d_sched_e != nullptr && max_tiles <= g_ws.sched_cap) return;
  if (g_ws.d_sched_e)  cudaFree(g_ws.d_sched_e);
  if (g_ws.d_sched_tm) cudaFree(g_ws.d_sched_tm);
  if (g_ws.h_sched_e)  cudaFreeHost(g_ws.h_sched_e);
  if (g_ws.h_sched_tm) cudaFreeHost(g_ws.h_sched_tm);
  cudaMalloc(&g_ws.d_sched_e,  max_tiles * sizeof(int32_t));
  cudaMalloc(&g_ws.d_sched_tm, max_tiles * sizeof(int32_t));
  cudaMallocHost(&g_ws.h_sched_e,  max_tiles * sizeof(int32_t));
  cudaMallocHost(&g_ws.h_sched_tm, max_tiles * sizeof(int32_t));
  g_ws.sched_cap = max_tiles;
}

static void ensure_per_token(int T) {
  if (T <= g_ws.cap_T) return;
  if (g_ws.d_topk_idx)    cudaFree(g_ws.d_topk_idx);
  if (g_ws.d_topk_w)      cudaFree(g_ws.d_topk_w);
  if (g_ws.d_perm_token)  cudaFree(g_ws.d_perm_token);
  if (g_ws.d_perm_weight) cudaFree(g_ws.d_perm_weight);
  if (g_ws.d_A_fp8)       cudaFree(g_ws.d_A_fp8);
  if (g_ws.d_A_scale)     cudaFree(g_ws.d_A_scale);
  if (g_ws.d_G1)          cudaFree(g_ws.d_G1);
  if (g_ws.d_C_fp32)      cudaFree(g_ws.d_C_fp32);
  if (g_ws.d_out_f32)     cudaFree(g_ws.d_out_f32);

  // Per-expert Tk ≤ T (each token picks each expert at most once).
  // Total N_sel ≤ T * TOP_K.
  size_t nsmax = (size_t)T * TOP_K;

  cudaMalloc(&g_ws.d_topk_idx,    (size_t)T * TOP_K * sizeof(int32_t));
  cudaMalloc(&g_ws.d_topk_w,      (size_t)T * TOP_K * sizeof(float));
  cudaMalloc(&g_ws.d_perm_token,  nsmax * sizeof(int32_t));
  cudaMalloc(&g_ws.d_perm_weight, nsmax * sizeof(float));
  cudaMalloc(&g_ws.d_A_fp8,       nsmax * H  * sizeof(__nv_fp8_e4m3));
  cudaMalloc(&g_ws.d_A_scale,     nsmax * NH_BLKS * sizeof(float));
  cudaMalloc(&g_ws.d_G1,          nsmax * II * sizeof(float));
  cudaMalloc(&g_ws.d_C_fp32,      nsmax * I  * sizeof(float));
  cudaMalloc(&g_ws.d_out_f32,     (size_t)T * H * sizeof(float));
  g_ws.cap_T = T;
}

static PFN_cuTensorMapEncodeTiled_v12000 get_encode_tiled() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (fn != nullptr) return fn;
  void* ptr = nullptr;
  cudaDriverEntryPointQueryResult status;
  cudaError_t err = cudaGetDriverEntryPointByVersion(
      "cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &status);
  if (err == cudaSuccess && status == cudaDriverEntryPointSuccess) {
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }
  return fn;
}

static bool make_gemm1_tma_maps(
    const __nv_fp8_e4m3* A_base, int n_sel,
    const __nv_fp8_e4m3* W_base,
    CUtensorMap* A_tma, CUtensorMap* W_tma) {
  auto encode = get_encode_tiled();
  if (encode == nullptr || n_sel < BM) return false;

  const cuuint64_t a_dims[2] = {(cuuint64_t)H, (cuuint64_t)n_sel};
  const cuuint64_t a_strides[1] = {(cuuint64_t)H * sizeof(__nv_fp8_e4m3)};
  const cuuint32_t a_box[2] = {(cuuint32_t)BK_WSP, (cuuint32_t)BM};
  const cuuint32_t a_elem_stride[2] = {1, 1};
  CUresult ar = encode(
      A_tma, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
      const_cast<__nv_fp8_e4m3*>(A_base),
      a_dims, a_strides, a_box, a_elem_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (ar != CUDA_SUCCESS) return false;

  const cuuint64_t w_dims[2] = {
      (cuuint64_t)H,
      (cuuint64_t)E_LOCAL * (cuuint64_t)II};
  const cuuint64_t w_strides[1] = {
      (cuuint64_t)H * sizeof(__nv_fp8_e4m3)};
  const cuuint32_t w_box[2] = {(cuuint32_t)BK_WSP, (cuuint32_t)BN_WSP};
  const cuuint32_t w_elem_stride[2] = {1, 1};
  CUresult wr = encode(
      W_tma, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
      const_cast<__nv_fp8_e4m3*>(W_base),
      w_dims, w_strides, w_box, w_elem_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return wr == CUDA_SUCCESS;
}

}  // namespace

// ======================================================================
// Host entry.
// ======================================================================
using tvm::ffi::TensorView;

static void run_impl(
    TensorView routing_logits,       // [T, 256] fp32
    TensorView routing_bias,         // [256] bf16
    TensorView hidden_states,        // [T, 7168] fp8_e4m3
    TensorView hidden_states_scale,  // [56, T] fp32
    TensorView gemm1_weights,        // [32, 4096, 7168] fp8_e4m3
    TensorView gemm1_weights_scale,  // [32, 32, 56] fp32
    TensorView gemm2_weights,        // [32, 7168, 2048] fp8_e4m3
    TensorView gemm2_weights_scale,  // [32, 56, 16] fp32
    int64_t    local_expert_offset,
    double     routed_scaling_factor,
    TensorView output)               // [T, 7168] bf16
{
  const int T = (int)routing_logits.shape()[0];
  if (T == 0) return;
  const int leo   = (int)local_expert_offset;
  const float sf  = (float)routed_scaling_factor;

  cudaStream_t stream = 0;

  ensure_singletons();
  ensure_per_token(T);
  ensure_schedule(T);

  static bool gemm_shmem_attr_set = false;
  if (!gemm_shmem_attr_set) {
    cudaFuncSetAttribute(gemm_bf16_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)GEMM_SMEM_BYTES);
    cudaFuncSetAttribute(gemm_bf16_fp8w_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)GEMM_SMEM_BYTES_V2);
    cudaFuncSetAttribute(gemm_bf16_fp8w_grouped_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)GEMM_SMEM_BYTES_GRP);
    cudaFuncSetAttribute(gemm_bf16_fp8w_fused_acc_grouped_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)GEMM_SMEM_BYTES_GRP);
    cudaFuncSetAttribute(gemm_fp8_fp8_grouped_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)GEMM_FP8_SMEM_BYTES);
    cudaFuncSetAttribute(gemm_fp8_fp8_wsp_grouped_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)GEMM_WSP_SMEM_BYTES);
    cudaFuncSetAttribute(gemm_fp32_fp8w_fused_acc_grouped_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)GEMM_G2_SMEM_BYTES);
    gemm_shmem_attr_set = true;
  }

  const float*         p_logits  = static_cast<const float*>(routing_logits.data_ptr());
  const __nv_bfloat16* p_bias    = static_cast<const __nv_bfloat16*>(routing_bias.data_ptr());
  const __nv_fp8_e4m3* p_hs_fp8  = static_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr());
  const float*         p_hs_sc   = static_cast<const float*>(hidden_states_scale.data_ptr());
  const __nv_fp8_e4m3* p_w13_fp8 = static_cast<const __nv_fp8_e4m3*>(gemm1_weights.data_ptr());
  const float*         p_w13_sc  = static_cast<const float*>(gemm1_weights_scale.data_ptr());
  const __nv_fp8_e4m3* p_w2_fp8  = static_cast<const __nv_fp8_e4m3*>(gemm2_weights.data_ptr());
  const float*         p_w2_sc   = static_cast<const float*>(gemm2_weights_scale.data_ptr());
  __nv_bfloat16*       p_out     = static_cast<__nv_bfloat16*>(output.data_ptr());

  // 1. routing + local expert counts
  cudaMemsetAsync(g_ws.d_counts, 0, E_LOCAL * sizeof(int32_t), stream);
  route_kernel<<<T, E_GLOBAL, 0, stream>>>(
      p_logits, p_bias, sf, T, leo,
      g_ws.d_topk_idx, g_ws.d_topk_w, g_ws.d_counts);

  // 2. scan / fill
  scan_offsets_kernel<<<1, 1, 0, stream>>>(
      g_ws.d_counts, g_ws.d_offs, g_ws.d_write_ptr, g_ws.d_n_sel);
  {
    int th = 256;
    perm_fill_kernel<<<(T + th - 1) / th, th, 0, stream>>>(
        g_ws.d_topk_idx, g_ws.d_topk_w, T, leo,
        g_ws.d_write_ptr, g_ws.d_perm_token, g_ws.d_perm_weight);
  }

  // 3. Read offs to host (blocks until perm done).
  int32_t offs_host[E_LOCAL + 1];
  cudaMemcpyAsync(offs_host, g_ws.d_offs, (E_LOCAL + 1) * sizeof(int32_t),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  const int N_sel = offs_host[E_LOCAL];

  // 5. Gather permuted fp8 hidden states + per-row per-kblock scale (for FP8 MMA GEMM1).
  if (N_sel > 0) {
    {
      // 128 threads × 16B = 2048 bytes/block; H=7168 → ceil(7168/2048)=4 tiles.
      constexpr int GATHER_THREADS = 128;
      constexpr int GATHER_PER_BLK = GATHER_THREADS * 16;       // 2048 bytes
      constexpr int GATHER_TILES   = (H + GATHER_PER_BLK - 1) / GATHER_PER_BLK;
      dim3 grid(N_sel, GATHER_TILES);
      gather_hs_fp8_kernel<<<grid, GATHER_THREADS, 0, stream>>>(
          p_hs_fp8, p_hs_sc, g_ws.d_perm_token, T, N_sel,
          g_ws.d_A_fp8, g_ws.d_A_scale);
    }
  }

  // 7. zero fp32 accumulator (cudaMemsetAsync is ~6× faster than a naive 4B/thread kernel)
  cudaMemsetAsync(g_ws.d_out_f32, 0, (size_t)T * H * sizeof(float), stream);

  // Build grouped-GEMM schedules on host. Full 128-row tiles go first so
  // GEMM1 can use the warp-specialized bulk-copy path; tail tiles remain on
  // the proven non-WSP path for correct zero-fill boundary behavior.
  int total_m_tiles = 0;
  int total_full_m_tiles = 0;
  for (int e = 0; e < E_LOCAL; e++) {
    int Tk = offs_host[e + 1] - offs_host[e];
    int nmt_full = Tk / BM;
    for (int m = 0; m < nmt_full; m++) {
      g_ws.h_sched_e[total_m_tiles]  = e;
      g_ws.h_sched_tm[total_m_tiles] = m;
      total_m_tiles++;
    }
    total_full_m_tiles = total_m_tiles;
  }
  for (int e = 0; e < E_LOCAL; e++) {
    int Tk = offs_host[e + 1] - offs_host[e];
    int nmt_full = Tk / BM;
    if (Tk > nmt_full * BM) {
      g_ws.h_sched_e[total_m_tiles]  = e;
      g_ws.h_sched_tm[total_m_tiles] = nmt_full;
      total_m_tiles++;
    }
  }
  if (total_m_tiles > 0) {
    cudaMemcpyAsync(g_ws.d_sched_e,  g_ws.h_sched_e,
                    total_m_tiles * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_ws.d_sched_tm, g_ws.h_sched_tm,
                    total_m_tiles * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    // 8a. Grouped GEMM1 — single-path bf16-MMA kernel (Triton-precision).
    // WSP TMA path is disabled this round: its fp8-MMA partial rounding diverged
    // from Triton's bf16-MMA path and failed correctness on larger-T workloads.
    {
      dim3 grid((II + BN - 1) / BN, total_m_tiles);
      gemm_fp8_fp8_grouped_kernel<<<grid, THREADS_FP8, GEMM_FP8_SMEM_BYTES, stream>>>(
          g_ws.d_A_fp8, g_ws.d_A_scale,
          p_w13_fp8, p_w13_sc,
          g_ws.d_offs, g_ws.d_sched_e, g_ws.d_sched_tm,
          II, H, NII_BLKS, NH_BLKS,
          g_ws.d_G1, II);
    }
    // 8b. SwiGLU over all N_sel rows (fp32 → fp32).
    {
      int th = 128;
      dim3 grid((I / 4 + th - 1) / th, N_sel);
      swiglu_kernel<<<grid, th, 0, stream>>>(g_ws.d_G1, N_sel, g_ws.d_C_fp32);
    }
    // 8c. Grouped GEMM2: fp32 C × fp8 W (bf16x2 emulation) + fused weighted scatter.
    {
      dim3 grid((H + BN - 1) / BN, total_m_tiles);
      gemm_fp32_fp8w_fused_acc_grouped_kernel<<<grid, THREADS_FP8, GEMM_G2_SMEM_BYTES, stream>>>(
          g_ws.d_C_fp32, p_w2_fp8, p_w2_sc,
          g_ws.d_offs, g_ws.d_sched_e, g_ws.d_sched_tm,
          g_ws.d_perm_token, g_ws.d_perm_weight,
          H, I, NH_BLKS, NI_BLKS,
          g_ws.d_out_f32, H);
    }
  }

  // 9. cast fp32 -> bf16 output (vectorized, 4 elems per thread)
  {
    const int H4 = H / 4;  // H=7168 divisible by 4
    int th = 128;
    dim3 grid((H4 + th - 1) / th, T);
    fin_cast_kernel<<<grid, th, 0, stream>>>(g_ws.d_out_f32, T, p_out);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run_impl);
