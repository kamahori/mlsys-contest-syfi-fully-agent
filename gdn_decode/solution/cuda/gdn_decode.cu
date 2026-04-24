/*
 * GDN decode — JIT-compiled GPU kernel via NVRTC + CUDA Driver API.
 *
 * Why: this build env links my .so against libcudart.so.13 but the installed
 * driver is R570 / CUDA 12.8, so every CUDA Runtime API call (incl. <<<>>>
 * launches) returns error 35. Round 3 fixed correctness with a host-only
 * implementation (~2.4× over the PyTorch ref). To go faster we need actual
 * GPU compute. NVRTC and the Driver API both live below libcudart and have no
 * runtime-vs-driver version check, so JIT'ing PTX through libnvrtc.so.12 +
 * launching through libcuda.so.1 sidesteps the broken runtime entirely.
 *
 * Pipeline:
 *   1. dlopen libcuda.so.1 (driver) and libnvrtc.so.12 (CUDA 12.8 NVRTC).
 *   2. nvrtcCompileProgram on the embedded kernel source → PTX.
 *   3. cuModuleLoadData(PTX) — driver JITs PTX to sm_100 SASS.
 *   4. cuModuleGetFunction → CUfunction handle, cached for the process.
 *   5. cuLaunchKernel each call on the FFI env stream.
 *
 * If any step fails, we fall back to a correct host-side reference
 * (slow but verified — that was the round-3 implementation).
 *
 * Kernel design (V_TILE=8, 1 warp, K-warp-reduce — proven layout):
 *   Grid:  (B * Hv * V/V_TILE) = (B * 8 * 16)
 *   Block: 32 threads = 1 warp. Each warp owns 8 v-rows; each thread holds
 *          K_LOCAL = K/32 = 4 K-slots. State[8][4]=32 fp32 in registers.
 *   K-reductions are warp-shfl. State reads are float4-vectorized; new_state
 *   writes are float4-vectorized; output writes are scalar bf16.
 */

#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

// ===== Driver / NVRTC API forward decls (subset; ABI-compatible) ============
typedef int           CUresult;
typedef void*         CUstream;
typedef void*         CUmodule;
typedef void*         CUfunction;
typedef unsigned long long CUdeviceptr;
typedef int           nvrtcResult;
typedef void*         nvrtcProgram;

struct DrvAPI {
  CUresult (*cuInit)(unsigned);
  CUresult (*cuCtxGetCurrent)(void**);
  CUresult (*cuModuleLoadData)(CUmodule*, const void*);
  CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
  CUresult (*cuLaunchKernel)(CUfunction,
      unsigned, unsigned, unsigned,
      unsigned, unsigned, unsigned,
      unsigned, CUstream, void**, void**);
  CUresult (*cuMemcpyDtoH_v2)(void*, CUdeviceptr, size_t);
  CUresult (*cuMemcpyHtoD_v2)(CUdeviceptr, const void*, size_t);
  CUresult (*cuMemsetD8Async)(CUdeviceptr, unsigned char, size_t, CUstream);
  CUresult (*cuFuncSetAttribute)(CUfunction, int, int);
};

struct NvrtcAPI {
  nvrtcResult (*nvrtcCreateProgram)(nvrtcProgram*, const char*, const char*,
                                    int, const char* const*, const char* const*);
  nvrtcResult (*nvrtcCompileProgram)(nvrtcProgram, int, const char* const*);
  nvrtcResult (*nvrtcGetPTXSize)(nvrtcProgram, size_t*);
  nvrtcResult (*nvrtcGetPTX)(nvrtcProgram, char*);
  nvrtcResult (*nvrtcGetProgramLogSize)(nvrtcProgram, size_t*);
  nvrtcResult (*nvrtcGetProgramLog)(nvrtcProgram, char*);
  nvrtcResult (*nvrtcDestroyProgram)(nvrtcProgram*);
  const char* (*nvrtcGetErrorString)(nvrtcResult);
};

static DrvAPI       g_drv   = {};
static NvrtcAPI     g_nvrtc = {};
static CUmodule     g_module = nullptr;
static CUfunction   g_kernel = nullptr;
static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;
static bool         g_gpu_ok = false;

// ===== Embedded kernel source (compiled by NVRTC at first call) =============
static const char* KERNEL_SRC = R"CUDA(
#include <cuda_bf16.h>

#define K_DIM 128
#define V_DIM 128
#define Hq 4
#define Hv 8
#define V_TILE 8
#define WARPS_PER_BLOCK 1
#define V_PER_WARP 8
#define N_V 16
#define BLOCK_THREADS 32
#define K_LOCAL 4

__device__ __forceinline__ float warp_reduce_sum(float v) {
  v += __shfl_xor_sync(0xffffffff, v, 16);
  v += __shfl_xor_sync(0xffffffff, v,  8);
  v += __shfl_xor_sync(0xffffffff, v,  4);
  v += __shfl_xor_sync(0xffffffff, v,  2);
  v += __shfl_xor_sync(0xffffffff, v,  1);
  return v;
}

extern "C" __global__ __launch_bounds__(BLOCK_THREADS, 12)
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
  const int v_tile_idx  = pid - bh * N_V;
  const int b_idx       = bh / Hv;
  const int h_idx       = bh - b_idx * Hv;
  const int qh          = h_idx >> 1;          // (h*Hq)/Hv with GVA=2
  const int v_base      = v_tile_idx * V_TILE;

  const int tid     = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane    = tid & 31;
  const int k_lo    = lane * K_LOCAL;
  const int v_warp  = v_base + warp_id * V_PER_WARP;

  // ---- gates ------------------------------------------------------------
  const float A_log    = A_log_ptr[h_idx];
  const float dt_bias  = dt_bias_ptr[h_idx];
  const float a_val    = __bfloat162float(a_ptr[b_idx * Hv + h_idx]);
  const float b_val    = __bfloat162float(b_ptr[b_idx * Hv + h_idx]);
  const float x        = a_val + dt_bias;
  const float softplus = (x > 20.f) ? x : log1pf(expf(x));
  const float g        = expf(-expf(A_log) * softplus);
  const float beta     = 1.f / (1.f + expf(-b_val));

  // ---- q, k -------------------------------------------------------------
  const int qk_base = b_idx * Hq * K_DIM + qh * K_DIM + k_lo;
  float q_arr[K_LOCAL], k_arr[K_LOCAL];
  #pragma unroll
  for (int i = 0; i < K_LOCAL; ++i) {
    q_arr[i] = __bfloat162float(q_ptr[qk_base + i]);
    k_arr[i] = __bfloat162float(k_ptr[qk_base + i]);
  }

  // ---- v slab -----------------------------------------------------------
  float v_mine = 0.f;
  if (lane < V_PER_WARP) {
    v_mine = __bfloat162float(v_ptr[b_idx * Hv * V_DIM + h_idx * V_DIM + v_warp + lane]);
  }

  // ---- state slab into registers ---------------------------------------
  const float* s_bh = state_ptr + (b_idx * Hv + h_idx) * V_DIM * K_DIM;
  float S[V_PER_WARP][K_LOCAL];
  #pragma unroll
  for (int jt = 0; jt < V_PER_WARP; ++jt) {
    const int row_off = (v_warp + jt) * K_DIM + k_lo;
    const float4 v4 = *reinterpret_cast<const float4*>(s_bh + row_off);
    S[jt][0] = v4.x; S[jt][1] = v4.y; S[jt][2] = v4.z; S[jt][3] = v4.w;
  }

  // ---- kS[jt] = sum_k k[k] * S[jt,k] -----------------------------------
  float kS[V_PER_WARP];
  #pragma unroll
  for (int jt = 0; jt < V_PER_WARP; ++jt) {
    float local = 0.f;
    #pragma unroll
    for (int s = 0; s < K_LOCAL; ++s) local += k_arr[s] * S[jt][s];
    kS[jt] = warp_reduce_sum(local);
  }

  // ---- update S -> S_new in regs + write new_state (float4) ------------
  float* ns_bh = new_state + (b_idx * Hv + h_idx) * V_DIM * K_DIM;
  #pragma unroll
  for (int jt = 0; jt < V_PER_WARP; ++jt) {
    const float v_jt = __shfl_sync(0xffffffff, v_mine, jt);
    const float r    = v_jt - g * kS[jt];
    float4 nv;
    nv.x = g * S[jt][0] + beta * k_arr[0] * r;
    nv.y = g * S[jt][1] + beta * k_arr[1] * r;
    nv.z = g * S[jt][2] + beta * k_arr[2] * r;
    nv.w = g * S[jt][3] + beta * k_arr[3] * r;
    S[jt][0] = nv.x; S[jt][1] = nv.y; S[jt][2] = nv.z; S[jt][3] = nv.w;
    *reinterpret_cast<float4*>(ns_bh + (v_warp + jt) * K_DIM + k_lo) = nv;
  }

  // ---- out[jt] = scale * sum_k q[k] * S_new[jt,k] ----------------------
  #pragma unroll
  for (int jt = 0; jt < V_PER_WARP; ++jt) {
    float local = 0.f;
    #pragma unroll
    for (int s = 0; s < K_LOCAL; ++s) local += q_arr[s] * S[jt][s];
    const float o = scale * warp_reduce_sum(local);
    if (lane == jt) {
      out_ptr[b_idx * Hv * V_DIM + h_idx * V_DIM + v_warp + jt] = __float2bfloat16(o);
    }
  }
}
)CUDA";

// ===== One-shot init: dlopen libs, JIT kernel, cache CUfunction =============
static void init_gpu_impl() {
  // Driver: libcuda is provided by the kernel driver itself, never has the
  // libcudart-vs-driver mismatch.
  void* libcuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (!libcuda) libcuda = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
  if (!libcuda) return;

  g_drv.cuInit               = (CUresult(*)(unsigned))                            dlsym(libcuda, "cuInit");
  g_drv.cuCtxGetCurrent      = (CUresult(*)(void**))                              dlsym(libcuda, "cuCtxGetCurrent");
  g_drv.cuModuleLoadData     = (CUresult(*)(CUmodule*, const void*))              dlsym(libcuda, "cuModuleLoadData");
  g_drv.cuModuleGetFunction  = (CUresult(*)(CUfunction*, CUmodule, const char*))  dlsym(libcuda, "cuModuleGetFunction");
  g_drv.cuLaunchKernel       = (CUresult(*)(CUfunction, unsigned, unsigned, unsigned,
                                            unsigned, unsigned, unsigned,
                                            unsigned, CUstream, void**, void**))  dlsym(libcuda, "cuLaunchKernel");
  g_drv.cuMemcpyDtoH_v2      = (CUresult(*)(void*, CUdeviceptr, size_t))          dlsym(libcuda, "cuMemcpyDtoH_v2");
  g_drv.cuMemcpyHtoD_v2      = (CUresult(*)(CUdeviceptr, const void*, size_t))    dlsym(libcuda, "cuMemcpyHtoD_v2");
  g_drv.cuMemsetD8Async      = (CUresult(*)(CUdeviceptr, unsigned char, size_t, CUstream))
                                                                                  dlsym(libcuda, "cuMemsetD8Async");
  g_drv.cuFuncSetAttribute   = (CUresult(*)(CUfunction, int, int))                dlsym(libcuda, "cuFuncSetAttribute");
  if (!g_drv.cuLaunchKernel || !g_drv.cuModuleLoadData || !g_drv.cuModuleGetFunction) return;
  if (g_drv.cuInit) g_drv.cuInit(0);

  // NVRTC: prefer the venv-bundled CUDA-12.8 build (matches the driver).
  void* libnvrtc = dlopen(
      "/raid/keisuke/flashinfer-contest/full-agent-pipeline/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
      RTLD_NOW | RTLD_GLOBAL);
  if (!libnvrtc) libnvrtc = dlopen("libnvrtc.so.12", RTLD_NOW | RTLD_GLOBAL);
  if (!libnvrtc) libnvrtc = dlopen("libnvrtc.so",    RTLD_NOW | RTLD_GLOBAL);
  if (!libnvrtc) return;

  g_nvrtc.nvrtcCreateProgram     = (nvrtcResult(*)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*))
                                   dlsym(libnvrtc, "nvrtcCreateProgram");
  g_nvrtc.nvrtcCompileProgram    = (nvrtcResult(*)(nvrtcProgram, int, const char* const*))      dlsym(libnvrtc, "nvrtcCompileProgram");
  g_nvrtc.nvrtcGetPTXSize        = (nvrtcResult(*)(nvrtcProgram, size_t*))                       dlsym(libnvrtc, "nvrtcGetPTXSize");
  g_nvrtc.nvrtcGetPTX            = (nvrtcResult(*)(nvrtcProgram, char*))                         dlsym(libnvrtc, "nvrtcGetPTX");
  g_nvrtc.nvrtcGetProgramLogSize = (nvrtcResult(*)(nvrtcProgram, size_t*))                       dlsym(libnvrtc, "nvrtcGetProgramLogSize");
  g_nvrtc.nvrtcGetProgramLog     = (nvrtcResult(*)(nvrtcProgram, char*))                         dlsym(libnvrtc, "nvrtcGetProgramLog");
  g_nvrtc.nvrtcDestroyProgram    = (nvrtcResult(*)(nvrtcProgram*))                               dlsym(libnvrtc, "nvrtcDestroyProgram");
  g_nvrtc.nvrtcGetErrorString    = (const char*(*)(nvrtcResult))                                 dlsym(libnvrtc, "nvrtcGetErrorString");
  if (!g_nvrtc.nvrtcCreateProgram || !g_nvrtc.nvrtcCompileProgram || !g_nvrtc.nvrtcGetPTX) return;

  // ---- Compile kernel -----------------------------------------------------
  nvrtcProgram prog = nullptr;
  if (g_nvrtc.nvrtcCreateProgram(&prog, KERNEL_SRC, "gdn_decode_kernel.cu",
                                 0, nullptr, nullptr) != 0) return;

  // Target compute_90 PTX — driver JITs to sm_100 on Blackwell.
  // Use compute_<n> (PTX), not sm_<n> (cubin), so the JIT can retarget.
  // NVRTC needs --include-path to find cuda_bf16.h (it ships its own copy
  // alongside the runtime headers); search a few well-known locations.
  std::vector<std::string> inc_dirs;
  const char* candidates[] = {
      "/raid/keisuke/flashinfer-contest/full-agent-pipeline/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/include",
      "/raid/keisuke/flashinfer-contest/flashinfer-bench-starter-kit/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/include",
      "/usr/local/cuda-12.8/include",
      "/usr/local/cuda/include",
  };
  for (const char* p : candidates) {
    if (access(p, R_OK) == 0) inc_dirs.push_back(std::string("--include-path=") + p);
  }

  std::vector<const char*> opts;
  opts.push_back("--gpu-architecture=compute_90");
  opts.push_back("--use_fast_math");
  opts.push_back("-default-device");
  opts.push_back("-std=c++17");
  for (auto& s : inc_dirs) opts.push_back(s.c_str());

  nvrtcResult cr = g_nvrtc.nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
  if (cr != 0) {
    size_t lsz = 0;
    g_nvrtc.nvrtcGetProgramLogSize(prog, &lsz);
    if (lsz > 1) {
      std::vector<char> log(lsz);
      g_nvrtc.nvrtcGetProgramLog(prog, log.data());
      fprintf(stderr, "[gdn_decode] NVRTC compile failed:\n%s\n", log.data());
    }
    g_nvrtc.nvrtcDestroyProgram(&prog);
    return;
  }

  size_t ptx_sz = 0;
  g_nvrtc.nvrtcGetPTXSize(prog, &ptx_sz);
  std::vector<char> ptx(ptx_sz);
  g_nvrtc.nvrtcGetPTX(prog, ptx.data());
  g_nvrtc.nvrtcDestroyProgram(&prog);

  if (g_drv.cuModuleLoadData(&g_module, ptx.data()) != 0) return;
  if (g_drv.cuModuleGetFunction(&g_kernel, g_module, "gdn_decode_kernel") != 0) return;

  g_gpu_ok = true;
}
static void init_gpu() { pthread_once(&g_init_once, init_gpu_impl); }

// ===== bf16 host helpers (for the host-fallback path) =======================
static inline float bf16_to_f(uint16_t b) {
  uint32_t u = ((uint32_t)b) << 16;
  float f; std::memcpy(&f, &u, sizeof(f));
  return f;
}
static inline uint16_t f_to_bf16(float f) {
  uint32_t u; std::memcpy(&u, &f, sizeof(u));
  if ((u & 0x7fffffff) > 0x7f800000) return (uint16_t)((u >> 16) | 0x0040);
  uint32_t rounded = u + 0x7fff + ((u >> 16) & 1);
  return (uint16_t)(rounded >> 16);
}

constexpr int K_DIM = 128;
constexpr int V_DIM = 128;
constexpr int Hq    = 4;
constexpr int Hv    = 8;
constexpr int V_TILE = 8;
constexpr int N_V    = V_DIM / V_TILE;            // 16
constexpr int BLOCK_THREADS = 32;

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

// ===== Host-side fallback (verified-correct round-3 implementation) =========
static void run_host(
    TensorView q, TensorView k, TensorView v,
    Optional<TensorView> state,
    TensorView A_log, TensorView a, TensorView dt_bias, TensorView b,
    float scale_f, TensorView output, TensorView new_state)
{
  const int64_t B = q.shape()[0];
  const size_t qk_sz    = (size_t)B * Hq * K_DIM;
  const size_t v_sz     = (size_t)B * Hv * V_DIM;
  const size_t state_sz = (size_t)B * Hv * V_DIM * K_DIM;
  const size_t ab_sz    = (size_t)B * Hv;

  std::vector<uint16_t> q_h(qk_sz), k_h(qk_sz), v_h(v_sz), a_h(ab_sz), b_h(ab_sz), out_h(v_sz);
  std::vector<float>    state_h(state_sz, 0.f), A_log_h(Hv), dt_bias_h(Hv), new_state_h(state_sz);

  auto D2H = [&](void* dst, const void* src, size_t bytes) {
    g_drv.cuMemcpyDtoH_v2(dst, (CUdeviceptr)(uintptr_t)src, bytes);
  };
  auto H2D = [&](void* dst, const void* src, size_t bytes) {
    g_drv.cuMemcpyHtoD_v2((CUdeviceptr)(uintptr_t)dst, src, bytes);
  };
  D2H(q_h.data(),       q.data_ptr(),       qk_sz * 2);
  D2H(k_h.data(),       k.data_ptr(),       qk_sz * 2);
  D2H(v_h.data(),       v.data_ptr(),       v_sz  * 2);
  D2H(a_h.data(),       a.data_ptr(),       ab_sz * 2);
  D2H(b_h.data(),       b.data_ptr(),       ab_sz * 2);
  D2H(A_log_h.data(),   A_log.data_ptr(),   Hv * 4);
  D2H(dt_bias_h.data(), dt_bias.data_ptr(), Hv * 4);
  if (state.has_value())
    D2H(state_h.data(), state.value().data_ptr(), state_sz * 4);

  for (int64_t bi = 0; bi < B; ++bi) {
    for (int h = 0; h < Hv; ++h) {
      const int qh = (h * Hq) / Hv;
      const float A_log_v = A_log_h[h];
      const float dt_b    = dt_bias_h[h];
      const float a_val   = bf16_to_f(a_h[bi * Hv + h]);
      const float b_val   = bf16_to_f(b_h[bi * Hv + h]);
      const float x       = a_val + dt_b;
      const float sp      = (x > 20.f) ? x : log1pf(expf(x));
      const float g_val   = expf(-expf(A_log_v) * sp);
      const float beta_v  = 1.f / (1.f + expf(-b_val));

      float q_vec[K_DIM], k_vec[K_DIM];
      for (int kk = 0; kk < K_DIM; ++kk) {
        q_vec[kk] = bf16_to_f(q_h[bi * Hq * K_DIM + qh * K_DIM + kk]);
        k_vec[kk] = bf16_to_f(k_h[bi * Hq * K_DIM + qh * K_DIM + kk]);
      }
      float qk_dot = 0.f;
      for (int kk = 0; kk < K_DIM; ++kk) qk_dot += q_vec[kk] * k_vec[kk];

      const float* state_bh = state_h.data() + (bi * Hv + h) * V_DIM * K_DIM;
      float* ns_bh          = new_state_h.data() + (bi * Hv + h) * V_DIM * K_DIM;
      for (int vv = 0; vv < V_DIM; ++vv) {
        float tk = 0.f, tq = 0.f;
        const float* row = state_bh + vv * K_DIM;
        for (int kk = 0; kk < K_DIM; ++kk) {
          tk += row[kk] * k_vec[kk];
          tq += row[kk] * q_vec[kk];
        }
        const float v_val = bf16_to_f(v_h[bi * Hv * V_DIM + h * V_DIM + vv]);
        const float delta = beta_v * (v_val - g_val * tk);
        const float out_v = scale_f * (g_val * tq + delta * qk_dot);
        out_h[bi * Hv * V_DIM + h * V_DIM + vv] = f_to_bf16(out_v);
        float* ns_row = ns_bh + vv * K_DIM;
        for (int kk = 0; kk < K_DIM; ++kk) {
          ns_row[kk] = g_val * row[kk] + delta * k_vec[kk];
        }
      }
    }
  }

  H2D(output.data_ptr(),    out_h.data(),       v_sz     * 2);
  H2D(new_state.data_ptr(), new_state_h.data(), state_sz * 4);
}

// ===== Entry point ==========================================================
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
  init_gpu();
  const int64_t B = q.shape()[0];
  const float scale_f =
      (scale == 0.0) ? (1.0f / sqrtf((float)K_DIM)) : (float)scale;

  if (!g_gpu_ok) {
    run_host(q, k, v, state, A_log, a, dt_bias, b, scale_f, output, new_state);
    return;
  }

  // ---- GPU launch via Driver API ------------------------------------------
  DLDevice dev = q.device();
  CUstream stream = static_cast<CUstream>(
      TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  void* state_ptr = nullptr;
  if (state.has_value()) {
    state_ptr = state.value().data_ptr();
  } else {
    g_drv.cuMemsetD8Async((CUdeviceptr)(uintptr_t)new_state.data_ptr(),
                          0, (size_t)B * Hv * V_DIM * K_DIM * sizeof(float),
                          stream);
    state_ptr = new_state.data_ptr();
  }

  void* q_ptr  = q.data_ptr();
  void* k_ptr  = k.data_ptr();
  void* v_ptr  = v.data_ptr();
  void* Al_ptr = A_log.data_ptr();
  void* a_ptr  = a.data_ptr();
  void* dt_ptr = dt_bias.data_ptr();
  void* b_ptr  = b.data_ptr();
  void* o_ptr  = output.data_ptr();
  void* ns_ptr = new_state.data_ptr();

  void* args[] = {
      (void*)&q_ptr, (void*)&k_ptr, (void*)&v_ptr, (void*)&state_ptr,
      (void*)&Al_ptr, (void*)&a_ptr, (void*)&dt_ptr, (void*)&b_ptr,
      (void*)&o_ptr, (void*)&ns_ptr, (void*)&scale_f
  };

  const unsigned grid  = (unsigned)(B * Hv * N_V);
  const unsigned block = (unsigned)BLOCK_THREADS;
  CUresult lr = g_drv.cuLaunchKernel(
      g_kernel,
      grid, 1, 1,
      block, 1, 1,
      0, stream, args, nullptr);
  if (lr != 0) {
    // Launch failed (rare — JIT'd OK but launch rejected). Fall back.
    run_host(q, k, v, state, A_log, a, dt_bias, b, scale_f, output, new_state);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_decode, run_impl);
