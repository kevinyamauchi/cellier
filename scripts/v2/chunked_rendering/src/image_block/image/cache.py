"""Backward-compatibility shim — import from image_block.core.cache instead."""

from __future__ import annotations

from image_block.core.cache import (
    CacheInfo,
    build_cache_texture,
)
from image_block.core.cache import (
    commit_block as commit_tile,  # old name preserved as alias
)
from image_block.core.cache import (
    compute_cache_info as _compute_cache_info,
)


def compute_cache_info(
    gpu_budget_bytes: int,
    block_size: int,
    overlap: int = 1,
    bytes_per_pixel: int = 4,
) -> CacheInfo:
    """Compute 2D cache dimensions (shim with old argument order).

    The core function uses ``(block_size, gpu_budget_bytes, ndim, ...)``.
    This shim preserves the old 2D signature ``(gpu_budget_bytes, block_size, ...)``.
    """
    return _compute_cache_info(
        block_size=block_size,
        gpu_budget_bytes=gpu_budget_bytes,
        ndim=2,
        overlap=overlap,
        bytes_per_element=bytes_per_pixel,
    )


__all__ = [
    "CacheInfo",
    "build_cache_texture",
    "commit_tile",
    "compute_cache_info",
]
