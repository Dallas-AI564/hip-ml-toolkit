# Multi-GPU Inference

## Tensor Parallelism

Split model layers across multiple GPUs.

```bash
# 2-way tensor parallel on MI300X
hipml serve meta-llama/Llama-3-70B --tensor-parallel 2

# 4-way on MI250X
hipml serve meta-llama/Llama-3-70B --tensor-parallel 4
```

## Pipeline Parallelism

Split model stages across GPUs.

```bash
hipml serve meta-llama/Llama-3-70B --pipeline-parallel 4
```

## RCCL (ROCm Collective Communications)

AMD's equivalent of NCCL for multi-GPU communication.

```bash
# All-reduce benchmark
mpirun -np 8 hipml benchmark --model meta-llama/Llama-3-70B --distributed
```

## Scaling Efficiency

| Config | GPUs | Throughput | Efficiency |
|--------|------|-----------|------------|
| MI300X × 1 | 1 | 743 tok/s | 100% |
| MI300X × 2 | 2 | 1,389 tok/s | 93% |
| MI300X × 4 | 4 | 2,614 tok/s | 88% |
| MI300X × 8 | 8 | 4,892 tok/s | 82% |
