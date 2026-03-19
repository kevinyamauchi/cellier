"""Unified GPU cache texture for 2D and 3D chunked rendering.

The cache is a pre-allocated texture containing a grid of block slots.
The total size is bounded by ``gpu_budget_bytes``.

Slot 0 is reserved (always zeros) — used as the destination for LUT
entries pointing to missing blocks.

Notes
-----
Axis order for ``gfx.Texture`` constructor matches numpy: ``(H, W)`` or
``(D, H, W)``.  However ``texture.update_range(offset, size)`` always
uses **(x, y, z)** order (reversed from numpy).
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
    ndim : int
        2 for a 2D tile cache, 3 for a 3D brick cache.
    grid_side : int
        Number of block slots per cache axis.
    cache_grid : tuple[int, ...]
        ``(grid_side,) * ndim``.
    cache_shape : tuple[int, ...]
        Pixel/voxel dimensions of the cache texture in numpy axis order.
        Length == ndim.
    n_slots : int
        Total number of slots (``grid_side ** ndim``).
    block_size : int
        Logical block side length (pixels for 2D, voxels for 3D).
    overlap : int
        Number of border elements duplicated on each side.
    padded_block_size : int
        ``block_size + 2 * overlap`` — actual elements per slot axis.
    """

    ndim: int
    grid_side: int
    cache_grid: tuple[int, ...]
    cache_shape: tuple[int, ...]
    n_slots: int
    block_size: int
    overlap: int
    padded_block_size: int


def compute_cache_info(
    block_size: int,
    gpu_budget_bytes: int,
    ndim: int = 3,
    overlap: int = 1,
    bytes_per_element: int = 4,
) -> CacheInfo:
    """Compute cache dimensions that fit within the GPU memory budget.

    Parameters
    ----------
    block_size : int
        Logical block side length (pixels for 2D, voxels for 3D).
    gpu_budget_bytes : int
        Maximum bytes for the cache texture.
    ndim : int
        2 for a tiled image cache, 3 for a brick volume cache.
    overlap : int
        Border elements duplicated on each side of every block.
    bytes_per_element : int
        Bytes per element (4 for float32).

    Returns
    -------
    info : CacheInfo
        Cache sizing metadata.
    """
    padded = block_size + 2 * overlap
    bytes_per_block = (padded**ndim) * bytes_per_element
    max_slots = gpu_budget_bytes // bytes_per_block
    grid_side = max(2, int(max_slots ** (1.0 / ndim)))
    n_slots = grid_side**ndim
    cache_grid = tuple(grid_side for _ in range(ndim))
    cache_shape = tuple(grid_side * padded for _ in range(ndim))
    return CacheInfo(
        ndim=ndim,
        grid_side=grid_side,
        cache_grid=cache_grid,
        cache_shape=cache_shape,
        n_slots=n_slots,
        block_size=block_size,
        overlap=overlap,
        padded_block_size=padded,
    )


def build_cache_texture(cache_info: CacheInfo) -> tuple[np.ndarray, gfx.Texture]:
    """Allocate the fixed-size cache texture (zeroed).

    Parameters
    ----------
    cache_info : CacheInfo
        Cache sizing metadata.

    Returns
    -------
    cache_data : np.ndarray
        The backing float32 array.
        ndim=2: shape ``(cH, cW)``; ndim=3: shape ``(cD, cH, cW)``.
    cache_tex : gfx.Texture
        pygfx texture wrapping ``cache_data``.
    """
    cache_data = np.zeros(cache_info.cache_shape, dtype=np.float32)
    cache_tex = gfx.Texture(cache_data, dim=cache_info.ndim)
    return cache_data, cache_tex


def commit_block(
    cache_data: np.ndarray,
    cache_tex: gfx.Texture,
    slot_grid_pos: tuple[int, ...],
    padded_block_size: int,
    data: np.ndarray,
) -> None:
    """Write one padded block into the CPU cache and schedule GPU upload.

    ``update_range`` always receives 3-element tuples even for dim=2
    textures:
      ndim=2: offset=(x, y, 0),  size=(w, h, 1)
      ndim=3: offset=(x, y, z),  size=(w, h, d)

    ``slot_grid_pos`` is in numpy axis order; this function converts to
    (x, y[, z]) before calling ``update_range``.

    Parameters
    ----------
    cache_data : np.ndarray
        The backing CPU array for the cache texture.
    cache_tex : gfx.Texture
        The pygfx texture to upload to.
    slot_grid_pos : tuple[int, ...]
        ``(sy, sx)`` for 2D or ``(sz, sy, sx)`` for 3D.
    padded_block_size : int
        Side length of the padded block (block_size + 2*overlap).
    data : np.ndarray
        Float32 block data of shape ``(pbs, pbs)`` or ``(pbs, pbs, pbs)``.
    """
    pbs = padded_block_size
    ndim = len(slot_grid_pos)

    if ndim == 2:
        sy, sx = slot_grid_pos
        y0 = sy * pbs
        x0 = sx * pbs
        cache_data[y0 : y0 + pbs, x0 : x0 + pbs] = data
        cache_tex.update_range(
            offset=(x0, y0, 0),
            size=(pbs, pbs, 1),
        )
    else:  # ndim == 3
        sz, sy, sx = slot_grid_pos
        z0 = sz * pbs
        y0 = sy * pbs
        x0 = sx * pbs
        cache_data[z0 : z0 + pbs, y0 : y0 + pbs, x0 : x0 + pbs] = data
        cache_tex.update_range(
            offset=(x0, y0, z0),
            size=(pbs, pbs, pbs),
        )
