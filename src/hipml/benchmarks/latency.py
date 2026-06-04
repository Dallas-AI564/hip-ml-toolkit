"""Latency benchmark — P50, P95, P99 measurement."""

import time
import statistics


def measure_latency(model_name: str, prompt: str, iterations: int = 100) -> dict:
    """Measure end-to-end inference latency."""
    try:
        from hipml.engines.vllm_engine import VLLMEngine
        engine = VLLMEngine(model_name).load()

        # Warmup
        for _ in range(10):
            engine.generate([prompt], max_tokens=32)

        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            engine.generate([prompt], max_tokens=1)
            latencies.append((time.perf_counter() - start) * 1000)

        latencies.sort()
        return {
            "model": model_name,
            "iterations": iterations,
            "p50_ms": round(statistics.median(latencies), 2),
            "p95_ms": round(latencies[int(len(latencies) * 0.95)], 2),
            "p99_ms": round(latencies[int(len(latencies) * 0.99)], 2),
            "mean_ms": round(statistics.mean(latencies), 2),
            "std_ms": round(statistics.stdev(latencies), 2),
        }
    except ImportError:
        return {"error": "vLLM not installed"}
