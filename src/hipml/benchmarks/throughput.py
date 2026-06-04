"""Throughput benchmark for LLM inference."""

import time
from typing import Optional


def run_throughput_benchmark(
    model_name: str,
    batch_sizes: list[int],
    seq_len: int = 2048,
    iterations: int = 100,
    warmup: int = 10,
) -> dict:
    """Run throughput benchmark across batch sizes."""
    prompt = "The future of artificial intelligence is" + " hello" * (seq_len // 10)
    results = []

    try:
        from hipml.engines.vllm_engine import VLLMEngine
        engine = VLLMEngine(model_name).load()

        for bs in batch_sizes:
            print(f"  Batch size {bs}...")

            # Warmup
            for _ in range(warmup):
                engine.generate([prompt] * bs, max_tokens=32)

            # Benchmark
            times = []
            token_counts = []
            for _ in range(iterations):
                start = time.perf_counter()
                outputs = engine.generate([prompt] * bs, max_tokens=128)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                token_counts.append(sum(len(o.split()) for o in outputs))

            avg_time = sum(times) / len(times)
            avg_tokens = sum(token_counts) / len(token_counts)

            results.append({
                "batch_size": bs,
                "avg_time_s": round(avg_time, 3),
                "tokens_per_second": round(avg_tokens / avg_time, 1),
                "p50_latency_ms": round(sorted(times)[len(times)//2] * 1000, 1),
                "p99_latency_ms": round(sorted(times)[int(len(times)*0.99)] * 1000, 1),
            })

    except ImportError:
        print("  vLLM not installed — running CPU mock benchmark")
        for bs in batch_sizes:
            results.append({
                "batch_size": bs,
                "avg_time_s": 0,
                "tokens_per_second": 0,
                "note": "Install vLLM for real benchmarks",
            })

    return {
        "model": model_name,
        "seq_len": seq_len,
        "iterations": iterations,
        "results": results,
    }
