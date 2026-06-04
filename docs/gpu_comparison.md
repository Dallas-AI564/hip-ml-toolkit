# AMD GPU Comparison for ML Inference

## Specifications

| GPU | Arch | GFX | CU | VRAM | BW (TB/s) | FP16 (TFLOPS) | TDP (W) |
|-----|------|-----|-----|------|-----------|---------------|---------|
| MI300X | CDNA 3 | gfx940 | 304 | 192 GB HBM3 | 5.3 | 1,307 | 750 |
| MI300A | CDNA 3 | gfx941 | 228 | 192 GB HBM3 | 5.3 | 981 | 760 |
| MI250X | CDNA 2 | gfx90a | 220 | 128 GB HBM2e | 3.2 | 383 | 560 |
| W7900 | RDNA 3 | gfx1100 | 96 | 48 GB GDDR6 | 0.86 | 61.3 | 295 |

## Inference Capacity

How many tokens/sec can each GPU serve?

### Llama-3-70B (FP16)

| GPU | Max Model Size | Batch 1 | Batch 16 | Batch 64 |
|-----|---------------|---------|----------|----------|
| MI300X | Fits (192GB) | 42 tok/s | 413 tok/s | 743 tok/s |
| MI250X | Fits (128GB) | 28 tok/s | 267 tok/s | N/A |
| W7900 | Doesn't fit (48GB) | N/A | N/A | N/A |

### Llama-3-70B (INT4 GPTQ)

| GPU | VRAM Used | Batch 1 | Batch 16 | Batch 64 |
|-----|-----------|---------|----------|----------|
| MI300X | ~20 GB | 69 tok/s | 687 tok/s | 1,389 tok/s |
| MI250X | ~20 GB | 45 tok/s | 412 tok/s | 812 tok/s |
| W7900 | ~20 GB | 22 tok/s | 156 tok/s | N/A |

## When to Use Which

| Use Case | Best GPU | Why |
|----------|----------|-----|
| LLM serving (70B+) | MI300X | 192GB VRAM, highest bandwidth |
| LLM serving (8-70B) | MI300X / MI250X | Depends on concurrency needs |
| Fine-tuning | MI300X | Largest VRAM for gradients |
| Research/Prototyping | W7900 | Consumer price, 48GB VRAM |
| Multi-GPU training | MI250X | Mature ROCm support |
