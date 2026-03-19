"""Backward-compatibility shim — import from image_block.core.cache instead."""

from __future__ import annotations

from image_block.core.cache import (
    CacheInfo,
    build_cache_texture,
)
from image_block.core.cache import (
    commit_block as commit_brick,  # old name preserved as alias
)
from image_block.core.cache import (
    compute_cache_info as _compute_cache_info,
)


def compute_cache_info(
    block_size: int,
    gpu_budget_bytes: int,
    overlap: int = 1,
    bytes_per_voxel: int = 4,
) -> CacheInfo:
    """Compute 3D cache dimensions (shim preserving old 3D argument order)."""
    return _compute_cache_info(
        block_size=block_size,
        gpu_budget_bytes=gpu_budget_bytes,
        ndim=3,
        overlap=overlap,
        bytes_per_element=bytes_per_voxel,
    )


__all__ = [
    "CacheInfo",
    "build_cache_texture",
    "commit_brick",
    "compute_cache_info",
]
