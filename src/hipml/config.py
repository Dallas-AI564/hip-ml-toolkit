"""YAML configuration loader."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load and validate config from YAML."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(p) as f:
        config = yaml.safe_load(f)

    return config


def get_default_config(gpu_name: str) -> dict:
    """Return sensible default config for a known GPU."""
    defaults = {
        "MI300X": _mi300x_config(),
        "MI250X": _mi250x_config(),
        "W7900": _w7900_config(),
    }
    for key, cfg in defaults.items():
        if key in gpu_name:
            return cfg
    return _generic_config()


def _mi300x_config() -> dict:
    return {
        "device": 0,
        "model": {"name": "meta-llama/Llama-3-70B", "precision": "fp8"},
        "serving": {"backend": "vllm", "max_batch_size": 64, "max_seq_len": 4096},
        "benchmark": {
            "batch_sizes": [1, 4, 8, 16, 32, 64],
            "sequence_lengths": [512, 1024, 2048, 4096],
            "iterations": 100, "warmup": 10,
        },
        "output": {"format": "json", "path": "results/mi300x/"},
    }


def _mi250x_config() -> dict:
    return {
        "device": 0,
        "model": {"name": "meta-llama/Llama-3-70B", "precision": "fp16"},
        "serving": {"backend": "vllm", "max_batch_size": 32, "max_seq_len": 2048},
        "benchmark": {
            "batch_sizes": [1, 4, 8, 16, 32],
            "sequence_lengths": [512, 1024, 2048],
            "iterations": 50, "warmup": 5,
        },
        "output": {"format": "json", "path": "results/mi250x/"},
    }


def _w7900_config() -> dict:
    return {
        "device": 0,
        "model": {"name": "meta-llama/Llama-3-8B", "precision": "fp16"},
        "serving": {"backend": "vllm", "max_batch_size": 8, "max_seq_len": 2048},
        "benchmark": {
            "batch_sizes": [1, 2, 4, 8],
            "sequence_lengths": [512, 1024, 2048],
            "iterations": 50, "warmup": 5,
        },
        "output": {"format": "json", "path": "results/w7900/"},
    }


def _generic_config() -> dict:
    return {
        "device": 0,
        "model": {"name": "meta-llama/Llama-3-8B", "precision": "fp16"},
        "serving": {"backend": "pytorch", "max_batch_size": 4},
        "benchmark": {"batch_sizes": [1, 2, 4], "sequence_lengths": [512, 1024]},
        "output": {"format": "json", "path": "results/generic/"},
    }
