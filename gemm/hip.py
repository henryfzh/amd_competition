#!POPCORN leaderboard amd-fp8-mm

# This is a submission template for popcorn leaderboard 'amd-fp8-mm'.
# Your task is as follows:
# > 
# > You will implement a custom fp8-blockwise matmul kernel optimized for MI300.
# > You will be given single-precision scaling factors for your matrices.
# > The shapes of all outer and inner dimensions of tensors are from DeepSeek-R1.
# > To be explicit, you will be given a tuple of tensors:
# > ```
# > (a, b, a_scale, b_scale, c)
# > ```
# > where `a` and `b` are the input matrices, `a_scale` and `b_scale` are the scaling factors for `a` and `b` respectively,
# > and `c` is the output matrix:
# > * `a` is M x K in column-major order in e4m3fnuz
# > * `b` is N x K in column-major order in e4m3fnuz
# > * `a_scale` is M x K // 128 in column-major order in fp32
# > * `b_scale` is N // 128 x K // 128 in column-major order in fp32
# > * `c` is M x N in ROW-major order in bf16
# > 
# > Matrix sizes `m` and `n` are divisible by 64, `k` is divisible by 128.
# > 
# > The ranking criteria is the geometric mean of the benchmark results.
# > 
# > For the grand price, your kernel will be evaluated against the speed of light analysis
# > and the solution closest to the speed of light will be awarded the grand price.
# > ```
# > The speed of light analysis is:
# >  M       N       K     time[us]
# > 1024    1536    7168      8.63
# > 1024    4608    7168     25.89
# > 6144    1536    7168     51.78
# > 6144    4608    7168    155.30
# > 1024    7168     256      3.17
# > 6144    7168     256     17.27
# > ```
# The deadline for this leaderboard is 2025-05-27 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!

# This script provides a template for using load_inline to run a HIP kernel for
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
CPP_WRAPPER = """
void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c);
"""

CUDA_SRC = """
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>

constexpr int BM = 64;  // Block size for M dimension
constexpr int BN = 64;  // Block size for N dimension
constexpr int BK = 128; // Block size for K dimension
constexpr int TM = 4;   // Thread tile size for M
constexpr int TN = 4;   // Thread tile size for N
constexpr int NUM_THREADS = 256;

__global__ void custom_kernel(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, 
                              const float* as, const float* bs, 
                              __hip_bfloat16* c, int m, int n, int k) {
                   
    // Your implementation here
    __shared__ __hip_fp8_e4m3_fnuz a_shared[BM][BK];
    __shared__ __hip_fp8_e4m3_fnuz b_shared[BN][BK];
    __shared__ float a_scale_shared[BM];
    __shared__ float b_scale_shared[BN/128];

    int bx = blockIdx.x * BM;
    int by = blockIdx.y * BN;
    int tx = threadIdx.x;

    // Each thread processes TMxTN elements
    int row = bx + (tx / (BN/TN)) * TM;
    int col = by + (tx % (BN/TN)) * TN;

    float accum[TM][TN] = {0.0f};

    for (int kb = 0; kb < k; kb += BK) {
        // Load A block into shared memory with FP8 conversion
        #pragma unroll
        for (int i = 0; i < BM; i += NUM_THREADS) {
            int load_row = i + tx;
            if (load_row < BM) {
                const uint8_t* a_ptr = &a[(kb)*m + (bx + load_row)];
                a_shared[load_row][0] = *reinterpret_cast<const __hip_fp8_e4m3_fnuz*>(a_ptr);
            }
        }

        // Load B block into shared memory with FP8 conversion
        #pragma unroll
        for (int j = 0; j < BN; j += NUM_THREADS) {
            int load_col = j + tx;
            if (load_col < BN) {
                const uint8_t* b_ptr = &b[(kb)*n + (by + load_col)];
                b_shared[load_col][0] = *reinterpret_cast<const __hip_fp8_e4m3_fnuz*>(b_ptr);
            }
        }
        __syncthreads();

        // Load scales into shared memory
        if (tx < BM) {
            int scale_idx = kb/BK;
            a_scale_shared[tx] = as[bx + tx + scale_idx*m];
        }
        if (tx < BN/128) {
            int scale_idx = kb/BK;
            b_scale_shared[tx] = bs[(by/128) + tx + scale_idx*(n/128)];
        }
        __syncthreads();

        // Compute partial sums
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            __hip_fp8_e4m3_fnuz a_frag[TM];
            __hip_fp8_e4m3_fnuz b_frag[TN];

            // Load A fragment
            #pragma unroll
            for (int t = 0; t < TM; ++t)
                a_frag[t] = a_shared[row - bx + t][kk];
            
            // Load B fragment
            #pragma unroll
            for (int t = 0; t < TN; ++t)
                b_frag[t] = b_shared[col - by + t][kk];

            // Compute with scaling
            float scale = a_scale_shared[row - bx] * b_scale_shared[(col - by)/128];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += static_cast<float>(a_frag[i]) * 
                                  static_cast<float>(b_frag[j]) * scale;
                }
            }
        }
        __syncthreads();
    }

    // Store results as BFloat16
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            if ((row + i) < m && (col + j) < n) {
                c[(row + i)*n + (col + j)] = c10::BFloat16(accum[i][j]);
            }
        }
    }
}

void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c) {
    int m = a.size(0);
    int n = b.size(0);
    int k = a.size(1);
    dim3 grid((m + BM - 1)/BM, (n + BN - 1)/BN);
    dim3 block(NUM_THREADS);
    
    custom_kernel<<<grid, block, 0, 0>>>(a.data_ptr<__hip_fp8_e4m3_fnuz>(),
                                  b.data_ptr<__hip_fp8_e4m3_fnuz>(),
                                  as.data_ptr<float>(),
                                  bs.data_ptr<float>(),
                                  c.data_ptr<__hip_bfloat16>(),
                                  m, n, k);

    //C10_CUDA_CHECK(cudaGetLastError());
}
"""

import os
os.environ["CXX"] = "clang++"

module = load_inline(
    name='fp8_mm',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['fp8_mm'],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20"],
)


def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    module.fp8_mm(a, b, a_scale, b_scale, c)
    return c


