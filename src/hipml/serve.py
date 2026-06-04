"""Model serving with vLLM, ONNX Runtime, or PyTorch backend."""

from typing import Optional


def serve_model(
    model_name: str,
    port: int = 8000,
    quantize: str = "none",
    tensor_parallel: int = 1,
    backend: str = "vllm",
):
    """Serve an LLM for inference."""
    if backend == "vllm":
        _serve_vllm(model_name, port, quantize, tensor_parallel)
    elif backend == "onnx":
        _serve_onnx(model_name, port)
    elif backend == "pytorch":
        _serve_pytorch(model_name, port)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _serve_vllm(model_name: str, port: int, quantize: str, tp: int):
    """Serve via vLLM (recommended for AMD GPUs)."""
    try:
        from vllm import AsyncLLMEngine, AsyncEngineArgs
    except ImportError:
        print("vLLM not installed. Install with: pip install vllm")
        print("Or: pip install hip-ml-toolkit[serve]")
        return

    engine_args = AsyncEngineArgs(
        model=model_name,
        tensor_parallel_size=tp,
        quantization=quantize if quantize != "none" else None,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print(f"vLLM engine loaded: {model_name}")
    print(f"Listening on port {port}")

    # FastAPI server
    try:
        import uvicorn
        from fastapi import FastAPI
        from vllm.entrypoints.openai.api_server import create_app

        app = create_app(engine)
        uvicorn.run(app, host="0.0.0.0", port=port)
    except ImportError:
        print("FastAPI/uvicorn not installed. Install with: pip install fastapi uvicorn")


def _serve_onnx(model_name: str, port: int):
    """Serve via ONNX Runtime."""
    print(f"ONNX Runtime serving: {model_name}")
    print("TODO: Implement ONNX Runtime serving")


def _serve_pytorch(model_name: str, port: int):
    """Serve via PyTorch eager mode."""
    print(f"PyTorch serving: {model_name}")
    print("TODO: Implement PyTorch serving")
