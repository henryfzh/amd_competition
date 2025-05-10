import torch
import triton
import time
import numpy as np
from amd_fp8_mm import custom_kernel  

def generate_test_case(M, N, K, seed=42):
    torch.manual_seed(seed)
    
    a = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(torch.float8_e4m3fnuz)
    b = torch.randint(-128, 127, (N, K), dtype=torch.int8).to(torch.float8_e4m3fnuz)
    
    a_scale = torch.rand(M, K // 128, dtype=torch.float32).abs() + 0.01
    b_scale = torch.rand(N // 128, K // 128, dtype=torch.float32).abs() + 0.01
    
    c = torch.empty((M, N), dtype=torch.bfloat16)
    
    return (a, b, a_scale, b_scale, c)

def reference_fp8_matmul(a, b, a_scale, b_scale):
    a_f32 = a.to(torch.float32) * a_scale.repeat_interleave(128, dim=1)[:, :a.shape[1]]
    b_f32 = b.to(torch.float32) * b_scale.repeat_interleave(128, dim=0)[:b.shape[0], :].repeat_interleave(128, dim=1)[:, :b.shape[1]]
    
    return torch.matmul(a_f32, b_f32.T).to(torch.bfloat16)

def test_correctness():
    print("Running correctness tests...")
    test_cases = [
        (64, 64, 128),    
        (128, 128, 256),  
        (1024, 1024, 2048), 
        (1024, 1536, 7168), 
        (6144, 4608, 7168)   
    ]
    
    for M, N, K in test_cases:
        print(f"\nTesting M={M}, N={N}, K={K}")
        data = generate_test_case(M, N, K)
        
        custom_output = custom_kernel(data)
        
        ref_output = reference_fp8_matmul(*data[:4])
        
        diff = (custom_output - ref_output).abs()
        max_diff = diff.max()
        avg_diff = diff.mean()
        
        print(f"Max difference: {max_diff.item():.6f}")
        print(f"Avg difference: {avg_diff.item():.6f}")
        
        assert max_diff < 1.0, f"Correctness test failed for M={M}, N={N}, K={K}"

def benchmark():
    print("\nRunning performance benchmarks...")
    test_cases = [
        (1024, 1536, 7168),  
        (1024, 4608, 7168),  
        (6144, 1536, 7168),  
        (6144, 4608, 7168), 
        (1024, 7168, 256),  
        (6144, 7168, 256)    
    ]
    
    warmup = 5
    repeats = 10
    
    for M, N, K in test_cases:
        print(f"\nBenchmarking M={M}, N={N}, K={K}")
        data = generate_test_case(M, N, K)
        
        for _ in range(warmup):
            _ = custom_kernel(data)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(repeats):
            _ = custom_kernel(data)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / repeats * 1e6  
        
        print(f"Average time: {elapsed:.2f} us")
        
        flops = 2 * M * N * K  
        tflops = flops / (elapsed * 1e6)  
        print(f"Throughput: {tflops:.2f} TFLOPs")

def stress_test():
    print("\nRunning stress tests...")
    test_cases = [
        (64, 7168, 7168),   
        (7168, 64, 7168),    
        (4096, 4096, 16384),  
        (8192, 8192, 8192)   
    ]
    
    for M, N, K in test_cases:
        print(f"\nStress test M={M}, N={N}, K={K}")
        try:
            data = generate_test_case(M, N, K)
            output = custom_kernel(data)
            print("Test passed!")
        except Exception as e:
            print(f"Test failed with error: {str(e)}")

if __name__ == "__main__":
    test_correctness()
    benchmark()
    stress_test()
    print("\nAll tests completed!")