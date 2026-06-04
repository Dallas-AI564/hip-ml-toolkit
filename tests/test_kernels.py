"""Tests for HIP kernels."""

import pytest


class TestKernelFiles:
    def test_fused_attention_exists(self):
        from pathlib import Path
        kernel = Path(__file__).parent.parent / "src" / "hipml" / "kernels" / "fused_attention.cpp"
        assert kernel.exists()

    def test_quant_gemm_exists(self):
        from pathlib import Path
        kernel = Path(__file__).parent.parent / "src" / "hipml" / "kernels" / "quant_gemm.cpp"
        assert kernel.exists()

    def test_rms_norm_exists(self):
        from pathlib import Path
        kernel = Path(__file__).parent.parent / "src" / "hipml" / "kernels" / "rms_norm.cpp"
        assert kernel.exists()
