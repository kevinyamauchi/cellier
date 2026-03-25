from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygfx as gfx


@dataclass(frozen=True)
class BlockCacheParameters2D:
    """Metadata about the allocated 2D cache texture.

    Attributes
    ----------
    grid_side : int
        Number of tile slots per cache axis.
    n_slots : int
        Total number of slots (``grid_side ** 2``).
    padded_block_size : int
        ``block_size + 2 * overlap`` -- actual pixels per slot axis.
    overlap : int
        Number of border pixels duplicated on each side.
    """

    grid_side: int
    n_slots: int
    padded_block_size: int
    overlap: int


def compute_block_cache_parameters_2d(
    gpu_budget_bytes: int,
    block_size: int,
    overlap: int = 1,
    bytes_per_pixel: int = 4,
) -> BlockCacheParameters2D:
    """Compute cache dimensions that fit within the GPU memory budget.

    This is for 2D blocks.

    Parameters
    ----------
    gpu_budget_bytes : int
        Maximum byte budget for the cache texture.
    block_size : int
        Tile side length in pixels (logical, no overlap).
    overlap : int
        Border pixels duplicated on each side.
    bytes_per_pixel : int
        Bytes per pixel (4 for float32).

    Returns
    -------
    info : BlockCacheParameters2D
        Cache sizing metadata.
    """
    padded = block_size + 2 * overlap
    bytes_per_tile = padded * padded * bytes_per_pixel
    max_slots = gpu_budget_bytes // bytes_per_tile
    # 2D grid: grid_side^2 = n_slots
    grid_side = int(math.isqrt(max_slots))
    # Ensure at least 2 (slot 0 is reserved).
    grid_side = max(grid_side, 2)
    n_slots = grid_side * grid_side

    return BlockCacheParameters2D(
        grid_side=grid_side,
        n_slots=n_slots,
        padded_block_size=padded,
        overlap=overlap,
    )


def build_cache_texture_2d(
    cache_info: BlockCacheParameters2D,
) -> tuple[np.ndarray, gfx.Texture]:
    """Allocate the fixed-size 2D cache texture (zeroed).

    Parameters
    ----------
    cache_info : CacheInfo
        Cache sizing metadata.

    Returns
    -------
    cache_data : np.ndarray
        The backing float32 array ``(cH, cW)``.
    cache_tex : gfx.Texture
        pygfx 2D texture wrapping ``cache_data``.
    """
    pbs = cache_info.padded_block_size
    gs = cache_info.grid_side
    cache_shape = (gs * pbs, gs * pbs)
    cache_data = np.zeros(cache_shape, dtype=np.float32)
    cache_tex = gfx.Texture(cache_data, dim=2)
    return cache_data, cache_tex
