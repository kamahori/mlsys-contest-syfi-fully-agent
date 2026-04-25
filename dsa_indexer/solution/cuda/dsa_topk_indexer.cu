// DSA TopK Indexer FP8 — dsa_topk_indexer_fp8_h64_d128_topk2048_ps64
//
// K cache memory layout per page (8448 bytes = 64 * 132):
//   bytes [0 .. 64*128-1]:  fp8 data, token t at bytes [t*128 .. t*128+127]
//   bytes [64*128 .. 8447]: float32 scales, token t at bytes [64*128 + t*4 .. +3]

#include <tvm/ffi/tvm_ffi.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cfloat>

#define NUM_HEADS   64
#define HEAD_DIM    128
#define PAGE_SIZE   64
#define TOPK        2048
#define HEAD_DIM_SF 132

// Q-head chunk size: 8 heads per chunk → smem ~12.3 KB → ~18 blocks/SM
// D-chunk size 16: cache 16 K-values in registers, accumulate across all Q heads
// → K dequantization reduced from Q_CHUNK*HEAD_DIM to HEAD_DIM per chunk (8x)
#define Q_CHUNK    8
#define N_Q_CHUNKS (NUM_HEADS / Q_CHUNK)   // 8
#define D_CHUNK    16                       // dims per inner tile
#define N_D_CHUNKS (HEAD_DIM / D_CHUNK)    // 8

__device__ __forceinline__ float fp8e4m3_to_float(uint8_t x) {
    __nv_fp8_e4m3 v;
    __builtin_memcpy(&v, &x, 1);
    return (float)v;
}

// ---- Kernel 1 v6: vectorized K transpose load (int4 = 16 bytes/thread/iter)
// Grid : (batch_size, max_num_pages)   Block: (PAGE_SIZE=64)
// Smem : ~12.3 KB  → ~18 blocks/SM
// vs v4: K global load uses 16-byte vector loads (8x fewer load instructions)
__global__ void __launch_bounds__(PAGE_SIZE, 18)
compute_scores_v6(
    const uint8_t* __restrict__ q_fp8,
    const uint8_t* __restrict__ k_cache,
    const float*   __restrict__ weights,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table,
    float*         __restrict__ scores,
    int batch_size, int max_num_pages, int max_seq_len
) {
    // K in d-major layout: k_fp8_T[d][t] = k_fp8_T[d*PAGE_SIZE+t]
    __shared__ uint8_t k_fp8_T[HEAD_DIM * PAGE_SIZE];   // 8192 B
    __shared__ float   k_scale_sh[PAGE_SIZE];            //  256 B
    __shared__ float   q_sh[Q_CHUNK * HEAD_DIM];         // 4096 B  (reused per chunk)
    __shared__ float   w_sh[Q_CHUNK];                    //   32 B
    // Total: 12576 B ≈ 12.3 KB

    const int b           = blockIdx.x;
    const int page_in_seq = blockIdx.y;
    const int t           = threadIdx.x;   // [0, 64)

    if (b >= batch_size) return;

    const int seq_len     = seq_lens[b];
    const int num_pages_b = (seq_len + PAGE_SIZE - 1) / PAGE_SIZE;

    if (page_in_seq >= num_pages_b) return;

    // Load K page into smem in transposed (d-major) layout
    // Use 16-byte vector loads (int4) to reduce instruction count 16x
    const int32_t page_idx = block_table[b * max_num_pages + page_in_seq];
    const uint8_t* k_page  = k_cache + (int64_t)page_idx * PAGE_SIZE * HEAD_DIM_SF;

    // Thread t loads its 128 fp8 K bytes using 8 × int4 (16-byte) loads
    const uint8_t* k_src = k_page + t * HEAD_DIM;
    #pragma unroll 8
    for (int d16 = 0; d16 < HEAD_DIM / 16; d16++) {
        // Load 16 bytes at once from global memory
        uint4 v;
        __builtin_memcpy(&v, k_src + d16 * 16, 16);
        // Scatter 16 bytes into transposed smem positions
        uint8_t b0[16];
        __builtin_memcpy(b0, &v, 16);
        #pragma unroll 16
        for (int dd = 0; dd < 16; dd++)
            k_fp8_T[(d16 * 16 + dd) * PAGE_SIZE + t] = b0[dd];
    }

    float sc;
    __builtin_memcpy(&sc, k_page + PAGE_SIZE * HEAD_DIM + t * 4, 4);
    k_scale_sh[t] = sc;

    __syncthreads();

    const int global_token = page_in_seq * PAGE_SIZE + t;
    const bool valid       = (global_token < seq_len);
    const float my_k_scale = k_scale_sh[t];
    float score            = 0.0f;

    const uint8_t* q_b = q_fp8 + (int64_t)b * NUM_HEADS * HEAD_DIM;

    #pragma unroll 1
    for (int chunk = 0; chunk < N_Q_CHUNKS; chunk++) {
        const int base_h = chunk * Q_CHUNK;

        // Cooperatively load Q chunk (coalesced)
        const uint8_t* q_src = q_b + base_h * HEAD_DIM;
        #pragma unroll 16
        for (int i = t; i < Q_CHUNK * HEAD_DIM; i += PAGE_SIZE)
            q_sh[i] = fp8e4m3_to_float(q_src[i]);
        if (t < Q_CHUNK)
            w_sh[t] = weights[b * NUM_HEADS + base_h + t];
        __syncthreads();

        if (valid) {
            float dot_acc[Q_CHUNK];
            #pragma unroll 8
            for (int h = 0; h < Q_CHUNK; h++) dot_acc[h] = 0.0f;

            #pragma unroll 8
            for (int db = 0; db < N_D_CHUNKS; db++) {
                const int d0 = db * D_CHUNK;

                float k_d[D_CHUNK];
                #pragma unroll 16
                for (int dd = 0; dd < D_CHUNK; dd++)
                    k_d[dd] = fp8e4m3_to_float(k_fp8_T[(d0 + dd) * PAGE_SIZE + t]) * my_k_scale;

                #pragma unroll 8
                for (int h = 0; h < Q_CHUNK; h++) {
                    const float* qh = q_sh + h * HEAD_DIM + d0;
                    #pragma unroll 16
                    for (int dd = 0; dd < D_CHUNK; dd++)
                        dot_acc[h] += qh[dd] * k_d[dd];
                }
            }

            #pragma unroll 8
            for (int h = 0; h < Q_CHUNK; h++)
                score += fmaxf(dot_acc[h], 0.0f) * w_sh[h];
        }
        __syncthreads();
    }

    if (valid)
        scores[(int64_t)b * max_seq_len + global_token] = score;
}

// ---- Kernel 2: bitonic sort + topk selection --------------------------
// Grid : (batch_size,)   Block: (256,)
// Dynamic smem: sort_n * 8 bytes
#define SORT_THREADS 256

__global__ void topk_select_kernel(
    const float*   __restrict__ scores,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table,
    int32_t*       __restrict__ topk_indices,
    int batch_size, int max_num_pages, int max_seq_len, int sort_n
) {
    extern __shared__ char smem[];
    float*   sh_vals = (float*)smem;
    int32_t* sh_idxs = (int32_t*)(smem + sort_n * sizeof(float));

    const int b   = blockIdx.x;
    const int tid = threadIdx.x;

    if (b >= batch_size) return;

    const int seq_len     = seq_lens[b];
    const int actual_topk = (seq_len < TOPK) ? seq_len : TOPK;

    for (int i = tid; i < TOPK; i += SORT_THREADS)
        topk_indices[b * TOPK + i] = -1;
    if (seq_len == 0) return;

    const float* sc_b = scores + (int64_t)b * max_seq_len;

    for (int i = tid; i < sort_n; i += SORT_THREADS) {
        sh_vals[i] = (i < seq_len) ? sc_b[i] : -FLT_MAX;
        sh_idxs[i] = (i < seq_len) ? i : -1;
    }
    __syncthreads();

    for (int size = 2; size <= sort_n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < sort_n / 2; i += SORT_THREADS) {
                int lo = (i / stride) * (2 * stride) + (i % stride);
                int hi = lo + stride;
                bool want_lo_greater = ((lo & size) == 0);
                float vlo = sh_vals[lo], vhi = sh_vals[hi];
                bool do_swap = want_lo_greater ? (vlo < vhi) : (vlo > vhi);
                if (do_swap) {
                    sh_vals[lo] = vhi; sh_vals[hi] = vlo;
                    int32_t ti = sh_idxs[lo];
                    sh_idxs[lo] = sh_idxs[hi]; sh_idxs[hi] = ti;
                }
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < actual_topk; i += SORT_THREADS) {
        int32_t tok = sh_idxs[i];
        if (tok < 0) { topk_indices[b * TOPK + i] = -1; continue; }
        int page_in_seq = tok / PAGE_SIZE;
        int offset      = tok % PAGE_SIZE;
        int32_t pg = block_table[b * max_num_pages + page_in_seq];
        topk_indices[b * TOPK + i] = pg * PAGE_SIZE + offset;
    }
}

// ---- TVM-FFI host function --------------------------------------------

void kernel_impl(
    tvm::ffi::Tensor q_index_fp8,
    tvm::ffi::Tensor k_index_cache_fp8,
    tvm::ffi::Tensor weights,
    tvm::ffi::Tensor seq_lens,
    tvm::ffi::Tensor block_table,
    tvm::ffi::Tensor topk_indices
) {
    const int batch_size    = (int)q_index_fp8.size(0);
    const int max_num_pages = (int)block_table.size(1);
    const int max_seq_len   = max_num_pages * PAGE_SIZE;

    const uint8_t* q_ptr   = (const uint8_t*)q_index_fp8.data_ptr();
    const uint8_t* k_ptr   = (const uint8_t*)k_index_cache_fp8.data_ptr();
    const float*   w_ptr   = (const float*)weights.data_ptr();
    const int32_t* sl_ptr  = (const int32_t*)seq_lens.data_ptr();
    const int32_t* bt_ptr  = (const int32_t*)block_table.data_ptr();
    int32_t*       out_ptr = (int32_t*)topk_indices.data_ptr();

    float* sc_ptr = nullptr;
    size_t sc_bytes = (size_t)batch_size * max_seq_len * sizeof(float);
    cudaMalloc(&sc_ptr, sc_bytes);

    cudaStream_t stream = 0;

    {
        dim3 grid(batch_size, max_num_pages);
        dim3 block(PAGE_SIZE);
        compute_scores_v6<<<grid, block, 0, stream>>>(
            q_ptr, k_ptr, w_ptr, sl_ptr, bt_ptr, sc_ptr,
            batch_size, max_num_pages, max_seq_len);
    }

    {
        int sort_n = 1;
        while (sort_n < max_seq_len) sort_n <<= 1;

        size_t smem = (size_t)sort_n * (sizeof(float) + sizeof(int32_t));

        static bool attr_set = false;
        if (!attr_set) {
            cudaFuncSetAttribute(topk_select_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
            attr_set = true;
        }

        topk_select_kernel<<<dim3(batch_size), dim3(SORT_THREADS), smem, stream>>>(
            sc_ptr, sl_ptr, bt_ptr, out_ptr,
            batch_size, max_num_pages, max_seq_len, sort_n);
    }

    cudaFree(sc_ptr);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, kernel_impl)
