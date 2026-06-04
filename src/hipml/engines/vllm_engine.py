"""vLLM inference engine for AMD GPUs."""

from typing import Optional


class VLLMEngine:
    """vLLM-based inference engine optimized for ROCm."""

    def __init__(self, model_name: str, tensor_parallel: int = 1,
                 quantization: Optional[str] = None, gpu_memory_util: float = 0.90):
        self.model_name = model_name
        self.tensor_parallel = tensor_parallel
        self.quantization = quantization
        self.gpu_memory_util = gpu_memory_util
        self._engine = None

    def load(self):
        """Load model into vLLM engine."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM not installed. pip install vllm")

        self._engine = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel,
            quantization=self.quantization,
            gpu_memory_utilization=self.gpu_memory_util,
            max_model_len=4096,
            dtype="auto",
            trust_remote_code=True,
        )
        return self

    def generate(self, prompts: list[str], max_tokens: int = 256,
                 temperature: float = 0.7) -> list[str]:
        """Generate text from prompts."""
        if self._engine is None:
            self.load()

        from vllm import SamplingParams
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
        )
        outputs = self._engine.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]

    def benchmark(self, prompt: str, batch_size: int = 1,
                  max_tokens: int = 128) -> dict:
        """Benchmark throughput for a given batch size."""
        import time

        prompts = [prompt] * batch_size
        start = time.perf_counter()
        outputs = self.generate(prompts, max_tokens=max_tokens)
        elapsed = time.perf_counter() - start

        total_tokens = sum(len(o.split()) for o in outputs)
        return {
            "batch_size": batch_size,
            "total_time_s": round(elapsed, 3),
            "tokens_per_second": round(total_tokens / elapsed, 1),
            "latency_per_token_ms": round(elapsed / total_tokens * 1000, 2),
        }
