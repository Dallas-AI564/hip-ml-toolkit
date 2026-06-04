# Quantization Guide for AMD GPUs

## Overview

Quantization reduces model size and increases inference speed by using
lower precision (FP8, INT4) instead of FP16/FP32.

## Methods

### FP8 (E4M3)
- **Best for:** MI300X (native FP8 support in CDNA 3)
- **Quality:** Minimal degradation vs FP16
- **Speed:** ~1.5-2x throughput improvement
- **VRAM:** 50% reduction vs FP16

```bash
hipml quantize --model meta-llama/Llama-3-70B --method fp8 --output llama-70b-fp8/
```

### GPTQ (INT4)
- **Best for:** All AMD GPUs
- **Quality:** Good with 128 group size
- **Speed:** ~2-3x throughput improvement
- **VRAM:** 75% reduction vs FP16

```bash
hipml quantize --model meta-llama/Llama-3-70B --method gptq --output llama-70b-gptq/
```

### AWQ (INT4)
- **Best for:** MI300X, MI250X
- **Quality:** Slightly better than GPTQ at same bits
- **Speed:** Similar to GPTQ
- **VRAM:** 75% reduction vs FP16

```bash
hipml quantize --model meta-llama/Llama-3-70B --method awq --output llama-70b-awq/
```

## Accuracy vs Speed Tradeoff

```
Quality (MMLU score)
  │
  │  FP16 ──────────────────────── 86.2
  │  FP8  ──────────────────────── 85.9  (99.7% of FP16)
  │  GPTQ INT4 ─────────────────── 84.1  (97.6% of FP16)
  │  AWQ INT4 ──────────────────── 84.5  (98.0% of FP16)
  │
  └──────────────────────────────── Throughput
       1x    1.5x    2x    2.5x   3x
```

## Recommendation

| GPU | Model Size | Recommended |
|-----|-----------|-------------|
| MI300X | ≤70B | FP8 |
| MI300X | 70B-405B | GPTQ INT4 |
| MI250X | ≤70B | GPTQ INT4 |
| W7900 | ≤13B | FP16 |
| W7900 | 13B-70B | GPTQ INT4 |
