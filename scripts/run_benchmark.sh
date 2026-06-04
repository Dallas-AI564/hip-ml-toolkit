#!/bin/bash
# run_benchmark.sh — Run full benchmark suite on AMD GPU

MODEL=${1:-meta-llama/Llama-3-8B}
BATCH_SIZES=${2:-"1,4,8,16"}
SEQ_LEN=${3:-2048}

echo "=== hip-ml-toolkit Benchmark ==="
echo "Model: $MODEL"
echo "Batch sizes: $BATCH_SIZES"
echo "Sequence length: $SEQ_LEN"
echo ""

# System info
echo "=== GPU Info ==="
rocm-smi --showproductname --showmeminfo vram 2>/dev/null || echo "rocm-smi not available"
echo ""

# Run benchmark
hipml benchmark \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZES" \
    --seq-len "$SEQ_LEN" \
    --iterations 100 \
    --output results/

echo ""
echo "=== Benchmark complete ==="
echo "Results: results/"
