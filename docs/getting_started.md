# Getting Started with hip-ml-toolkit

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| ROCm | 5.7 | 6.0+ |
| Python | 3.10 | 3.12 |
| GPU | Any AMD (gfx90a+) | MI300X |
| VRAM | 16 GB | 192 GB |
| Docker | 20.10 | Latest |

## Installation

### Option 1: pip

```bash
git clone https://github.com/Dallas-AI564/hip-ml-toolkit.git
cd hip-ml-toolkit
pip install -e ".[dev,serve]"
```

### Option 2: Docker (Recommended)

```bash
docker pull rocm/pytorch:latest
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video \
  -v $(pwd):/workspace hip-ml-toolkit
```

## Verify Installation

```bash
hipml info
```

Expected output (on MI300X):
```json
{
  "available": true,
  "name": "AMD Instinct MI300X",
  "architecture": "CDNA 3",
  "gfx": "gfx940",
  "vram_total_mb": 196608,
  "memory_bandwidth_gbps": 5300
}
```

## First Benchmark

```bash
hipml benchmark --model meta-llama/Llama-3-8B --batch-size 1,4,8 --seq-len 1024
```

## Serving a Model

```bash
hipml serve meta-llama/Llama-3-70B --quantize fp8 --port 8000

# Test with curl
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 50}'
```
