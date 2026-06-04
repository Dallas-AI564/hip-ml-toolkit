/*
 * silu_mul.cpp — Fused SiLU × Gate activation for AMD GPUs
 * SiLU(x) = x * sigmoid(x), then multiply with gate
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void silu_mul_kernel(
    const half* __restrict__ input,   // [N, 2, hidden]
    half* __restrict__ output,         // [N, hidden]
    int N, int hidden)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * hidden) return;

    int n = idx / hidden;
    int h = idx % hidden;

    float gate = __half2float(input[n * 2 * hidden + h]);
    float up = __half2float(input[n * 2 * hidden + hidden + h]);

    // SiLU(gate) * up
    float silu_gate = gate / (1.0f + expf(-gate));
    output[idx] = __float2half(silu_gate * up);
}

extern "C" void launch_silu_mul(
    const half* input, half* output, int N, int hidden,
    hipStream_t stream)
{
    int total = N * hidden;
    int block = 256;
    int grid = (total + block - 1) / block;
    silu_mul_kernel<<<grid, block, 0, stream>>>(input, output, N, hidden);
}
