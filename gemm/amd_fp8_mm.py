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
    # Pointers to matrices
    a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Matrix strides
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_ascale_m, stride_ascale_k,
    stride_bscale_n, stride_bscale_k,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    # Super-block in K dimension
    GROUP_M: tl.constexpr,
):
    """
    Compute: C = A @ B where:
    A is an [M, K] matrix in FP8 (E4M3FNUZ) with block-wise scaling
    B is an [N, K] matrix in FP8 (E4M3FNUZ) with block-wise scaling
    C is an [M, N] matrix in BF16
    """
    # -----------------------------------------------------------
    # Matrix multiplication with block-wise scaling for FP8
    # -----------------------------------------------------------

    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Each program handles a block of the output matrix
    # Use a grouped grid to improve parallelism
    group_id = pid_m // GROUP_M
    group_size = min(GROUP_M, (M + BLOCK_M - 1) // BLOCK_M - group_id * GROUP_M)
    pid_m = pid_m % GROUP_M
    
    # Block start indices
    block_m_idx = group_id * GROUP_M * BLOCK_M + pid_m * BLOCK_M
    block_n_idx = pid_n * BLOCK_N
    
    # Offsets in the M and N dimensions
    offs_m = block_m_idx + tl.arange(0, BLOCK_M)
    offs_n = block_n_idx + tl.arange(0, BLOCK_N)
    
    # Create M and N dimension masks
    m_mask = offs_m < M
    n_mask = offs_n < N
    
    # Pointers to output matrix C
    offs_cm = offs_m[:, None] * stride_cm
    offs_cn = offs_n[None, :] * stride_cn
    c_ptrs = c_ptr + offs_cm + offs_cn
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate through K dimension in blocks
    for k in range(0, K, BLOCK_K):
        # Block K dimension and mask
        offs_k = k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        
        # ----------------------------------------------------------------
        # Load and scale matrix A
        # ----------------------------------------------------------------
        # Calculate offsets for A matrix
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        
        # Load A block
        a_block = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Calculate the scale factor indices for A
        a_scale_block_idx = k // 128  # Each scale applies to 128 elements
        a_scale_ptrs = a_scale_ptr + offs_m[:, None] * stride_ascale_m + a_scale_block_idx * stride_ascale_k
        
        # Load and apply A scale factors
        a_scales = tl.load(a_scale_ptrs, mask=m_mask[:, None], other=0.0)
        # Convert A to FP32 and apply scaling
        a_fp32 = a_block.to(tl.float32) * a_scales
        
        # ----------------------------------------------------------------
        # Load and scale matrix B
        # ----------------------------------------------------------------
        # Calculate offsets for B matrix
        b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
        
        # Load B block
        b_block = tl.load(b_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Calculate the scale factor indices for B
        b_scale_block_n_idx = block_n_idx // 128  # N dimension scale block
        b_scale_block_k_idx = k // 128            # K dimension scale block
        b_scale_ptr_curr = b_scale_ptr + b_scale_block_n_idx * stride_bscale_n + b_scale_block_k_idx * stride_bscale_k
        
        # Load B scale factor (single scale for the entire block)
        b_scale = tl.load(b_scale_ptr_curr)
        
        # Convert B to FP32 and apply scaling
        b_fp32 = b_block.to(tl.float32) * b_scale
        
        # ----------------------------------------------------------------
        # Matrix multiplication
        # ----------------------------------------------------------------
        # Update accumulator with dot product
        acc += tl.dot(a_fp32, tl.trans(b_fp32))
    
    # Store result to C in BF16 format
    c_bf16 = acc.to(tl.bfloat16)
    tl.store(c_ptrs, c_bf16, mask=m_mask[:, None] & n_mask[None, :])

def custom_kernel(data: input_t) -> output_t:
    """
    Optimized implementation of block-scale fp8 gemm for MI300
    
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
    a, b, a_scale, b_scale, c = data
    
    # Get matrix dimensions
    M, K = a.shape
    N, _ = b.shape
    
    # Tuned block sizes for MI300 architecture
    # These should be tuned based on the compute capabilities of the MI300
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 128
    
    # Group size for increased parallelism
    GROUP_M = 8
    
    # Calculate grid dimensions
    grid = (triton.cdiv(M, BLOCK_M) * GROUP_M, triton.cdiv(N, BLOCK_N))
    
    # Launch kernel
    fp8_matmul_kernel[grid](
        a, b, a_scale, b_scale, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )
    
    return c