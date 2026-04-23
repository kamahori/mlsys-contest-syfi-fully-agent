/*
 * GDN decode — CUDA kernel, two-warp block layout.
 *
 * Grid: (B * Hv * V/V_TILE).  Block: 64 threads (two warps).
 *   warp 0 handles rows jt = 0..V_HALF-1  of the V-tile slab,
 *   warp 1 handles rows jt = V_HALF..V_TILE-1.
 * Each thread owns K_LOCAL = K/32 = 4 K-slots.  Reductions are intra-warp
 * __shfl (independent between warps — each warp has its own jt range).
 *
 * Rationale: single-warp blocks leave too much register state per thread
 * (S[16][4] = 64 fp32), capping occupancy at 25 % theoretical / ~21 %
 * achieved.  Halving S to S[V_HALF][4]=S[8][4] drops reg pressure enough
 * that the register limit opens up to ~14 blocks/SM × 2 warps = 28 warps/SM
 * (theoretical ~44 %), hiding HBM latency better on the memory-latency-bound
 * decode path.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

constexpr int K  = 128;
constexpr int V  = 128;
constexpr int Hq = 4;
constexpr int Hv = 8;
constexpr int V_TILE = 16;
constexpr int V_HALF = V_TILE / 2;           // 8
constexpr int N_V = V / V_TILE;              // 8
constexpr int WARPS_PER_BLOCK = 2;
constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * 32;  // 64
constexpr int K_LOCAL = K / 32;              // 4 (warp-local)

static __forceinline__ __device__ float warp_reduce_sum(float v) {
  v += __shfl_xor_sync(0xffffffff, v, 16);
  v += __shfl_xor_sync(0xffffffff, v, 8);
  v += __shfl_xor_sync(0xffffffff, v, 4);
  v += __shfl_xor_sync(0xffffffff, v, 2);
  v += __shfl_xor_sync(0xffffffff, v, 1);
  return v;
}

__global__ __launch_bounds__(BLOCK_THREADS, 12)
void gdn_decode_kernel(
    const __nv_bfloat16* __restrict__ q_ptr,
    const __nv_bfloat16* __restrict__ k_ptr,
    const __nv_bfloat16* __restrict__ v_ptr,
    const float*         __restrict__ state_ptr,
    const float*         __restrict__ A_log_ptr,
    const __nv_bfloat16* __restrict__ a_ptr,
    const float*         __restrict__ dt_bias_ptr,
    const __nv_bfloat16* __restrict__ b_ptr,
    __nv_bfloat16*       __restrict__ out_ptr,
    float*               __restrict__ new_state,
    float scale)
{
  const int pid         = blockIdx.x;
  const int bh          = pid / N_V;
  const int v_tile_idx  = pid % N_V;
  const int b_idx       = bh / Hv;
  const int h_idx       = bh % Hv;
  const int qh          = (h_idx * Hq) / Hv;
  const int v_base      = v_tile_idx * V_TILE;

  const int tid     = threadIdx.x;           // 0..63
  const int warp_id = tid >> 5;              // 0 or 1
  const int lane    = tid & 31;
  const int k_lo    = lane * K_LOCAL;        // 0, 4, 8, ..., 124 (per-warp)
  // Each warp handles V_HALF consecutive rows starting at v_warp:
  const int v_warp  = v_base + warp_id * V_HALF;

  // -------- gates --------------------------------------------------------
  const float A_log    = A_log_ptr[h_idx];
  const float dt_bias  = dt_bias_ptr[h_idx];
  const float a_val    = __bfloat162float(a_ptr[b_idx * Hv + h_idx]);
  const float b_val    = __bfloat162float(b_ptr[b_idx * Hv + h_idx]);
  const float x        = a_val + dt_bias;
  const float softplus = (x > 20.f) ? x : log1pf(expf(x));
  const float g        = expf(-expf(A_log) * softplus);
  const float beta     = 1.f / (1.f + expf(-b_val));

  // -------- q, k (K_LOCAL elements per thread) ---------------------------
  // Both warps read the same (qh) q/k vector — duplicate but cheap.
  const int qk_base = b_idx * Hq * K + qh * K;
  float k_arr[K_LOCAL], q_arr[K_LOCAL];
  #pragma unroll
  for (int i = 0; i < K_LOCAL; ++i) {
    k_arr[i] = __bfloat162float(k_ptr[qk_base + k_lo + i]);
    q_arr[i] = __bfloat162float(q_ptr[qk_base + k_lo + i]);
  }

  // -------- v slab: warp owns v[v_warp..v_warp+V_HALF] -------------------
  // lane<V_HALF loads its own v element for this warp's row range.
  float v_mine = 0.f;
  if (lane < V_HALF) {
    v_mine = __bfloat162float(
        v_ptr[b_idx * Hv * V + h_idx * V + v_warp + lane]);
  }

  // -------- state slab: S[V_HALF][K_LOCAL] in registers (half of before)
  const float* s_bh = state_ptr + (b_idx * Hv + h_idx) * V * K;
  static_assert(K_LOCAL == 4, "float4 load expects K_LOCAL == 4");
  float S[V_HALF][K_LOCAL];
  #pragma unroll
  for (int jt = 0; jt < V_HALF; ++jt) {
    const int row_base = (v_warp + jt) * K + k_lo;
    const float4 v4 = __ldg(reinterpret_cast<const float4*>(s_bh + row_base));
    S[jt][0] = v4.x; S[jt][1] = v4.y; S[jt][2] = v4.z; S[jt][3] = v4.w;
  }

  // -------- kS[jt] = Σ_i k[i] * S[i, jt] ---------------------------------
  float kS[V_HALF];
  #pragma unroll
  for (int jt = 0; jt < V_HALF; ++jt) {
    float local = 0.f;
    #pragma unroll
    for (int s = 0; s < K_LOCAL; ++s) {
      local += k_arr[s] * S[jt][s];
    }
    kS[jt] = warp_reduce_sum(local);   // every lane gets the full sum
  }

  // -------- In-place update: S <- S_new; also write new_state ------------
  float* ns_bh = new_state + (b_idx * Hv + h_idx) * V * K;
  #pragma unroll
  for (int jt = 0; jt < V_HALF; ++jt) {
    const float v_jt = __shfl_sync(0xffffffff, v_mine, jt);
    const float r = v_jt - g * kS[jt];
    float4 new_v4;
    new_v4.x = g * S[jt][0] + beta * k_arr[0] * r;
    new_v4.y = g * S[jt][1] + beta * k_arr[1] * r;
    new_v4.z = g * S[jt][2] + beta * k_arr[2] * r;
    new_v4.w = g * S[jt][3] + beta * k_arr[3] * r;
    S[jt][0] = new_v4.x; S[jt][1] = new_v4.y;
    S[jt][2] = new_v4.z; S[jt][3] = new_v4.w;
    *reinterpret_cast<float4*>(ns_bh + (v_warp + jt) * K + k_lo) = new_v4;
  }

  // -------- o[jt] = scale * Σ_i q[i] * S_new[i, jt] ----------------------
  // Lane `lane==jt` writes one output element for this warp's range.
  #pragma unroll
  for (int jt = 0; jt < V_HALF; ++jt) {
    float local = 0.f;
    #pragma unroll
    for (int s = 0; s < K_LOCAL; ++s) {
      local += q_arr[s] * S[jt][s];
    }
    const float o = scale * warp_reduce_sum(local);
    if (lane == jt) {
      out_ptr[b_idx * Hv * V + h_idx * V + v_warp + jt] =
          __float2bfloat16(o);
    }
  }
}

// -------------------------------------------------------------------------
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
    double scale,
    TensorView output,
    TensorView new_state)
{
  const int64_t B = q.shape()[0];
  float scale_f = (scale == 0.0) ? (1.0f / sqrtf((float)K)) : (float)scale;

  const float* state_data;
  if (state.has_value()) {
    state_data = static_cast<const float*>(state.value().data_ptr());
  } else {
    cudaMemsetAsync(new_state.data_ptr(), 0,
                    B * Hv * V * K * sizeof(float), 0);
    state_data = static_cast<const float*>(new_state.data_ptr());
  }

  dim3 grid(B * Hv * N_V);
  dim3 block(BLOCK_THREADS);

  gdn_decode_kernel<<<grid, block, 0, 0>>>(
      static_cast<const __nv_bfloat16*>(q.data_ptr()),
      static_cast<const __nv_bfloat16*>(k.data_ptr()),
      static_cast<const __nv_bfloat16*>(v.data_ptr()),
      state_data,
      static_cast<const float*>(A_log.data_ptr()),
      static_cast<const __nv_bfloat16*>(a.data_ptr()),
      static_cast<const float*>(dt_bias.data_ptr()),
      static_cast<const __nv_bfloat16*>(b.data_ptr()),
      static_cast<__nv_bfloat16*>(output.data_ptr()),
      static_cast<float*>(new_state.data_ptr()),
      scale_f);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_decode, run_impl);
