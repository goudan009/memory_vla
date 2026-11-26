"""CUDA reproducibility setup; no CUDA execution required for this contract."""
import os
import torch


def test_setup_enables_reproducible_kernels(monkeypatch):
    from v2_1_groot_memoryvla.distributed import setup
    monkeypatch.setenv('WORLD_SIZE','1')
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    previous=torch.are_deterministic_algorithms_enabled()
    try:
        setup()
        assert torch.are_deterministic_algorithms_enabled()
        assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    finally:
        torch.use_deterministic_algorithms(previous)
