"""Tests for inference engines."""

import pytest


class TestVLLMEngine:
    def test_import(self):
        """vLLM engine module should be importable."""
        from hipml.engines.vllm_engine import VLLMEngine
        assert VLLMEngine is not None

    def test_init(self):
        from hipml.engines.vllm_engine import VLLMEngine
        engine = VLLMEngine("test-model", tensor_parallel=1)
        assert engine.model_name == "test-model"
        assert engine.tensor_parallel == 1
