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

from task import input_t, output_t
import torch
import triton
import triton.language as tl

@triton.jit
def fp8_matmul_kernel(
    a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_ascale_m, stride_ascale_k,
    stride_bscale_n, stride_bscale_k,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        a_scale_block_k = k // BLOCK_K
        a_scale_off_m = offs_m[:, None] * stride_ascale_m
        a_scale_off_k = a_scale_block_k * stride_ascale_k
        a_scale_ptrs = a_scale_ptr + a_scale_off_m + a_scale_off_k
        a_scales = tl.load(a_scale_ptrs, mask=offs_m[:, None] < M, other=0.0)
        
        b_scale_block_n = (pid_n * BLOCK_N) // 128
        b_scale_block_k = a_scale_block_k
        b_scale_off_n = b_scale_block_n * stride_bscale_n
        b_scale_off_k = b_scale_block_k * stride_bscale_k
        b_scale_ptr_curr = b_scale_ptr + b_scale_off_n + b_scale_off_k
        b_scale = tl.load(b_scale_ptr_curr)
        
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        a_f32 = a.to(tl.float32) * a_scales
        
        b_ptrs = b_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        b = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        b_f32 = b.to(tl.float32) * b_scale
        
        acc += tl.dot(a_f32, tl.trans(b_f32), allow_tf32=False)
    
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c = acc.to(tl.bfloat16)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of block-scale fp8 gemm
    Args:
        data: Tuple that expands to:
            a: torch.Tensor[float8_e4m3fnuz] of shape [m, k],
            b: torch.Tensor[float8_e4m3fnuz] of shape [n, k],
            a_scale: torch.Tensor[float32] of shape [m, k // 128],
            b_scale: torch.Tensor[float32] of shape [n // 128, k // 128],
            c: torch.Tensor[bfloat16] of shape [m, n]
    Returns:
        Tensor containing output in bf16
    """
    # c: [m, n] is pre-allocated memory to avoid timing allocation overhead.
    a, b, a_scale, b_scale, c = data

    # Your implementation here
    M, K = a.shape
    N, _ = b.shape
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    fp8_matmul_kernel[grid](
        a, b, a_scale, b_scale, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return c

