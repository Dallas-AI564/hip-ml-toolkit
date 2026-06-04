/*
 * rotary_emb.cpp — Rotary Position Embedding (RoPE) for AMD GPUs
 * Vectorized sin/cos computation with HIP intrinsics
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

__global__ void rotary_embedding_kernel(
    half* __restrict__ query,       // [batch, seq, heads, head_dim]
    half* __restrict__ key,         // [batch, seq, kv_heads, head_dim]
    const half* __restrict__ cos,   // [seq, head_dim/2]
    const half* __restrict__ sin,   // [seq, head_dim/2]
    int batch_size, int seq_len, int num_heads, int head_dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * seq_len * num_heads * head_dim;

    if (idx >= total) return;

    int d = idx % head_dim;
    int half_dim = head_dim / 2;

    if (d >= head_dim) return;

    int pos = (idx / head_dim) % seq_len;
    int pair_idx = d % half_dim;

    float c = __half2float(cos[pos * half_dim + pair_idx]);
    float s = __half2float(sin[pos * half_dim + pair_idx]);

    float val = __half2float(query[idx]);
    float pair_val;
    if (d < half_dim) {
        pair_val = __half2float(query[idx + half_dim]);
        query[idx] = __float2half(val * c - pair_val * s);
    } else {
        pair_val = __half2float(query[idx - half_dim]);
        query[idx] = __float2half(val * c + pair_val * s);
    }
}

extern "C" void launch_rotary_embedding(
    half* query, half* key,
    const half* cos, const half* sin,
    int batch_size, int seq_len, int num_heads, int head_dim,
    hipStream_t stream)
{
    int total = batch_size * seq_len * num_heads * head_dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    rotary_embedding_kernel<<<grid, block, 0, stream>>>(
        query, key, cos, sin, batch_size, seq_len, num_heads, head_dim);
}
