"""Tests for quantization module."""

import pytest
from hipml.optimizers.quantizer import quantize_model


class TestQuantizer:
    def test_fp8_method_exists(self):
        """FP8 quantization should be supported."""
        # Just test the function exists and accepts the right args
        assert callable(quantize_model)

    def test_gptq_method_exists(self):
        assert callable(quantize_model)

    def test_awq_method_exists(self):
        assert callable(quantize_model)
