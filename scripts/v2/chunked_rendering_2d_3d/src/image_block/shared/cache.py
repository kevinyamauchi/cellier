"""Fixed-size GPU cache texture for tiled 2D image rendering.

The cache is a pre-allocated 2D texture containing a grid of tile slots.
The total size is bounded by ``gpu_budget_bytes``.

Each cache slot stores ``(block_size + 2*overlap)^2`` pixels.
Slot 0 is reserved (always zeros) -- used as the destination for LUT
entries pointing to missing tiles.

Notes
-----
Axis order for ``gfx.Texture`` constructor matches numpy: ``(H, W)``.
However ``texture.update_range(offset, size)`` uses **(W, H)** order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygfx as gfx


@dataclass(frozen=True)
class CacheInfo:
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


def compute_cache_info(
    gpu_budget_bytes: int,
    block_size: int,
    overlap: int = 1,
    bytes_per_pixel: int = 4,
) -> CacheInfo:
    """Compute cache dimensions that fit within the GPU memory budget.

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
    info : CacheInfo
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

    return CacheInfo(
        grid_side=grid_side,
        n_slots=n_slots,
        padded_block_size=padded,
        overlap=overlap,
    )


def build_cache_texture(cache_info: CacheInfo) -> tuple[np.ndarray, gfx.Texture]:
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


def commit_tile(
    cache_data: np.ndarray,
    cache_tex: gfx.Texture,
    slot_grid_pos: tuple[int, int],
    padded_block_size: int,
    data: np.ndarray,
) -> None:
    """Write a (padded) tile into the cache and schedule a GPU upload.

    Parameters
    ----------
    cache_data : np.ndarray
        The backing CPU array for the cache texture.
    cache_tex : gfx.Texture
        The pygfx texture to upload to.
    slot_grid_pos : tuple[int, int]
        ``(sy, sx)`` slot position in the cache grid.
    padded_block_size : int
        Side length of the padded tile (block_size + 2*overlap).
    data : np.ndarray
        Float32 tile data of shape ``(pbs, pbs)``.
    """
    sy, sx = slot_grid_pos
    pbs = padded_block_size
    y0 = sy * pbs
    x0 = sx * pbs
    cache_data[y0 : y0 + pbs, x0 : x0 + pbs] = data
    # pygfx update_range always requires 3D tuples (x, y, z) even for 2D textures.
    cache_tex.update_range(
        offset=(x0, y0, 0),
        size=(pbs, pbs, 1),
    )


if __name__ == "__main__":
    info = compute_cache_info(
        gpu_budget_bytes=64 * 1024 * 1024,
        block_size=32,
        overlap=1,
    )
    print(f"CacheInfo: grid_side={info.grid_side}, n_slots={info.n_slots}")
    print(f"  padded_block_size={info.padded_block_size}")
    pbs = info.padded_block_size
    tex_h = info.grid_side * pbs
    tex_bytes = tex_h * tex_h * 4
    print(f"  cache texture: {tex_h}x{tex_h} = {tex_bytes / 1024**2:.1f} MB")
    print("CacheInfo OK")
