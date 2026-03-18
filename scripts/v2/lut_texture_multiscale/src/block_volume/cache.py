"""Fixed-size GPU cache texture for bricked volume rendering.

Phase 1: the cache is a pre-allocated 3D texture containing a grid of
brick slots.  The total size is bounded by ``gpu_budget_bytes``.

Each cache slot stores ``(block_size + 2*overlap)³`` voxels: the
payload plus a 1-voxel border duplicated from spatial neighbours.
This prevents hardware linear interpolation from bleeding across
unrelated cache slots at brick boundaries.

Slot 0 is reserved (always zeros) — used as the destination for LUT
entries pointing to out-of-bounds or missing bricks.

Notes
-----
Axis order for ``gfx.Texture`` constructor matches numpy: ``(D, H, W)``.
However ``texture.update_range(offset, size)`` uses **(W, H, D)** order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygfx as gfx


@dataclass(frozen=True)
class CacheInfo:
    """Metadata about the allocated cache texture.

    Attributes
    ----------
    grid_side : int
        Number of brick slots per cache axis.
    cache_grid : tuple[int, int, int]
        ``(grid_side, grid_side, grid_side)``.
    cache_shape : tuple[int, int, int]
        Voxel dimensions ``(cD, cH, cW)`` of the cache texture.
    n_slots : int
        Total number of slots (``grid_side ** 3``).
    block_size : int
        *Logical* brick side length in voxels (payload, no overlap).
    overlap : int
        Number of border voxels duplicated on each side.
    padded_block_size : int
        ``block_size + 2 * overlap`` — actual voxels per slot axis.
    """

    grid_side: int
    cache_grid: tuple[int, int, int]
    cache_shape: tuple[int, int, int]
    n_slots: int
    block_size: int
    overlap: int
    padded_block_size: int


def compute_cache_info(
    block_size: int,
    gpu_budget_bytes: int,
    overlap: int = 1,
    bytes_per_voxel: int = 4,
) -> CacheInfo:
    """Compute cache dimensions that fit within the GPU memory budget.

    Parameters
    ----------
    block_size : int
        *Logical* brick side length in voxels.
    gpu_budget_bytes : int
        Maximum byte budget for the cache texture.
    overlap : int
        Border voxels duplicated on each side of every slot.
    bytes_per_voxel : int
        Bytes per voxel (4 for float32).

    Returns
    -------
    info : CacheInfo
        Cache sizing metadata.
    """
    padded = block_size + 2 * overlap
    bytes_per_brick = padded**3 * bytes_per_voxel
    max_slots = gpu_budget_bytes // bytes_per_brick
    grid_side = int(max_slots ** (1.0 / 3.0))
    # Ensure at least 2 (slot 0 is reserved).
    grid_side = max(grid_side, 2)

    cache_grid = (grid_side, grid_side, grid_side)
    cache_shape = (grid_side * padded,) * 3
    n_slots = grid_side**3

    return CacheInfo(
        grid_side=grid_side,
        cache_grid=cache_grid,
        cache_shape=cache_shape,
        n_slots=n_slots,
        block_size=block_size,
        overlap=overlap,
        padded_block_size=padded,
    )


def build_cache_texture(info: CacheInfo) -> tuple[np.ndarray, gfx.Texture]:
    """Allocate the fixed-size cache texture (zeroed).

    Parameters
    ----------
    info : CacheInfo
        Cache sizing metadata.

    Returns
    -------
    cache_data : np.ndarray
        The backing float32 array ``(cD, cH, cW)``.
    cache_tex : gfx.Texture
        pygfx 3D texture wrapping ``cache_data``.
    """
    cache_data = np.zeros(info.cache_shape, dtype=np.float32)
    cache_tex = gfx.Texture(cache_data, dim=3)
    return cache_data, cache_tex


def commit_brick(
    cache_data: np.ndarray,
    cache_tex: gfx.Texture,
    grid_pos: tuple[int, int, int],
    padded_block_size: int,
    data: np.ndarray,
) -> None:
    """Write a (padded) brick into the cache and schedule a GPU upload.

    Parameters
    ----------
    cache_data : np.ndarray
        The backing CPU array for the cache texture.
    cache_tex : gfx.Texture
        The pygfx texture to upload to.
    grid_pos : tuple[int, int, int]
        ``(sz, sy, sx)`` slot position in the cache grid.
    padded_block_size : int
        Side length of the padded brick (block_size + 2*overlap).
    data : np.ndarray
        Float32 brick data of shape ``(pbs, pbs, pbs)``.
    """
    sz, sy, sx = grid_pos
    pbs = padded_block_size
    z0 = sz * pbs
    y0 = sy * pbs
    x0 = sx * pbs
    cache_data[z0 : z0 + pbs, y0 : y0 + pbs, x0 : x0 + pbs] = data
    # pygfx update_range uses (W, H, D) = (x, y, z) order.
    cache_tex.update_range(
        offset=(x0, y0, z0),
        size=(pbs, pbs, pbs),
    )
