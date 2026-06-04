"""AMD GPU detection and capability reporting."""

import subprocess
import json
from typing import Optional


# Known AMD GPU specs
GPU_SPECS = {
    "MI300X": {
        "architecture": "CDNA 3", "gfx": "gfx940", "cu_count": 304,
        "vram_total_mb": 196608, "vram_type": "HBM3",
        "memory_bandwidth_gbps": 5300, "fp16_tflops": 1307,
        "fp32_tflops": 163.4, "matrix_core": "MFMA",
        "tdp_watts": 750,
    },
    "MI300A": {
        "architecture": "CDNA 3", "gfx": "gfx941", "cu_count": 228,
        "vram_total_mb": 196608, "vram_type": "HBM3",
        "memory_bandwidth_gbps": 5300, "fp16_tflops": 981,
        "fp32_tflops": 122.6, "matrix_core": "MFMA",
        "tdp_watts": 760,
    },
    "MI250X": {
        "architecture": "CDNA 2", "gfx": "gfx90a", "cu_count": 220,
        "vram_total_mb": 131072, "vram_type": "HBM2e",
        "memory_bandwidth_gbps": 3200, "fp16_tflops": 383,
        "fp32_tflops": 47.9, "matrix_core": "MFMA",
        "tdp_watts": 560,
    },
    "W7900": {
        "architecture": "RDNA 3", "gfx": "gfx1100", "cu_count": 96,
        "vram_total_mb": 49152, "vram_type": "GDDR6",
        "memory_bandwidth_gbps": 864, "fp16_tflops": 61.3,
        "fp32_tflops": 30.6, "matrix_core": "WMMA",
        "tdp_watts": 295,
    },
}


def _detect_rocm() -> Optional[dict]:
    """Detect AMD GPU via rocm-smi."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        return data
    except Exception:
        return None


def _detect_hip_smi() -> Optional[dict]:
    """Detect AMD GPU via hip-smi."""
    try:
        result = subprocess.run(
            ["hip-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        lines = result.stdout.strip().split("\n")
        if lines:
            parts = lines[0].split(", ")
            return {"name": parts[0].strip(), "vram_mb": int(parts[1].strip().split()[0])}
    except Exception:
        return None


def get_device_info(device_index: int = 0) -> dict:
    """Get comprehensive AMD GPU information."""
    info = _detect_rocm() or _detect_hip_smi()

    if info is None:
        return {
            "available": False,
            "error": "No AMD GPU detected. Ensure ROCm is installed.",
            "install_hint": "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/",
        }

    name = info.get("name", "Unknown")
    specs = None
    for key, val in GPU_SPECS.items():
        if key in name:
            specs = val
            break

    result = {
        "available": True,
        "name": name,
        "device_index": device_index,
    }

    if specs:
        result.update(specs)
        result["recognized"] = True
    else:
        result["recognized"] = False
        result["note"] = "GPU not in database. Add specs to device.py GPU_SPECS."

    return result


def get_gfx_arch(name: str) -> str:
    """Return GFX architecture string for a GPU name."""
    for key, val in GPU_SPECS.items():
        if key in name:
            return val["gfx"]
    return "gfx90a"  # generic fallback
