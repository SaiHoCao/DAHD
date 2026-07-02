"""Device and GPU utility functions for DAHD experiments."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GPUInfo:
    """Information about the current GPU device.

    Attributes:
        name: GPU device name (e.g., 'NVIDIA A100-SXM4-80GB').
        total_memory_gb: Total GPU memory in gigabytes.
        cuda_version: CUDA runtime version string.
        device_count: Number of available GPU devices.
        compute_capability: GPU compute capability (major, minor).
    """

    name: str
    total_memory_gb: float
    cuda_version: str
    device_count: int
    compute_capability: tuple[int, int]


def get_gpu_info(device_index: int = 0) -> GPUInfo:
    """Get information about the GPU device.

    Args:
        device_index: Index of the GPU device to query.

    Returns:
        GPUInfo dataclass with device details.

    Raises:
        RuntimeError: If CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")

    device_count = torch.cuda.device_count()
    if device_index >= device_count:
        raise ValueError(
            f"Device index {device_index} out of range. "
            f"Only {device_count} device(s) available."
        )

    props = torch.cuda.get_device_properties(device_index)
    total_memory_gb = props.total_mem / (1024 ** 3)
    cuda_version = torch.version.cuda or "unknown"
    compute_capability = (props.major, props.minor)

    return GPUInfo(
        name=props.name,
        total_memory_gb=round(total_memory_gb, 2),
        cuda_version=cuda_version,
        device_count=device_count,
        compute_capability=compute_capability,
    )


def setup_deterministic(seed: int = 42) -> None:
    """Set up deterministic execution for reproducibility.

    Configures random seeds for Python, NumPy, and PyTorch (CPU + CUDA).
    Also enables deterministic algorithms in PyTorch where possible.

    Args:
        seed: Random seed value to use across all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enable deterministic algorithms (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 1.8+ deterministic flag
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class MemoryUsage:
    """GPU memory usage snapshot.

    Attributes:
        allocated_gb: Currently allocated GPU memory in GB.
        reserved_gb: Currently reserved (cached) GPU memory in GB.
        max_allocated_gb: Peak allocated GPU memory since last reset, in GB.
        free_gb: Estimated free GPU memory in GB.
    """

    allocated_gb: float
    reserved_gb: float
    max_allocated_gb: float
    free_gb: float


def get_memory_usage(device_index: int = 0) -> MemoryUsage:
    """Get current GPU memory usage.

    Args:
        device_index: Index of the GPU device to query.

    Returns:
        MemoryUsage dataclass with memory statistics.

    Raises:
        RuntimeError: If CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")

    device = torch.device(f"cuda:{device_index}")

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    # Estimate free memory
    total = torch.cuda.get_device_properties(device_index).total_mem / (1024 ** 3)
    free = total - reserved

    return MemoryUsage(
        allocated_gb=round(allocated, 3),
        reserved_gb=round(reserved, 3),
        max_allocated_gb=round(max_allocated, 3),
        free_gb=round(free, 3),
    )


def reset_memory_stats(device_index: int = 0) -> None:
    """Reset peak memory statistics for the given device.

    Args:
        device_index: Index of the GPU device.
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_index}")
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
