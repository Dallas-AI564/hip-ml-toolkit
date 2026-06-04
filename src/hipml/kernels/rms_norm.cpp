/*
 * rms_norm.cpp — RMS Normalization for AMD GPUs
 * Optimized for 64-wide wavefronts (CDNA architecture)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define WARP_SIZE 64

__device__ __forceinline__ float wave_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__global__ void rms_norm_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    float epsilon,
    int hidden_size)
{
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += WARP_SIZE) {
        float val = __half2float(input[batch_idx * hidden_size + i]);
        sum_sq += val * val;
    }

    // Wavefront reduction
    sum_sq = wave_reduce_sum(sum_sq);

    // RMS
    float rms = rsqrtf(sum_sq / hidden_size + epsilon);

    // Normalize and scale
    for (int i = tid; i < hidden_size; i += WARP_SIZE) {
        float val = __half2float(input[batch_idx * hidden_size + i]);
        float w = __half2float(weight[i]);
        output[batch_idx * hidden_size + i] = __float2half(val * rms * w);
    }
}

extern "C" void launch_rms_norm(
    const half* input, const half* weight, half* output,
    float epsilon, int batch_size, int hidden_size,
    hipStream_t stream)
{
    rms_norm_kernel<<<batch_size, WARP_SIZE, 0, stream>>>(
        input, weight, output, epsilon, hidden_size);
}
