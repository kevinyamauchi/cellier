from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygfx as gfx


@dataclass(frozen=True)
class BlockCacheParameters3D:
    """Sizing metadata for the fixed-size 3-D GPU block cache.

    The cache is a single 3-D texture subdivided into a uniform cubic
    grid of brick slots.  ``grid_side`` is the only independent dimension
    parameter; all other fields are derived from it together with
    ``block_size`` and ``overlap``.

    Parameters
    ----------
    grid_side : int
        Number of slots along each axis of the cache grid.
        The cache holds ``grid_side ** 3`` slots in total.
        Slot 0 is reserved (always zero), leaving ``grid_side**3 - 1``
        usable slots.  Computed from the GPU budget by
        ``compute_cache_info()``.
    block_size : int
        Logical brick side length in voxels — the payload region,
        not counting the overlap border.
    overlap : int
        Number of voxels duplicated from neighbouring bricks on each
        side of every slot.  Prevents hardware linear interpolation
        from bleeding across slot boundaries.

    Attributes
    ----------
    padded_block_size : int
        ``block_size + 2 * overlap`` — the actual side length of one
        slot in the cache texture, in voxels.
    n_slots : int
        ``grid_side ** 3`` — total number of slots including the
        reserved slot 0.
    cache_shape : tuple[int, int, int]
        Shape of the backing numpy array and GPU texture in numpy axis
        order ``(cD, cH, cW)``, where each dimension equals
        ``grid_side * padded_block_size``.
        Pass directly to ``np.zeros()`` and ``gfx.Texture()``.
        Note: ``gfx.Texture.update_range()`` uses the reversed
        ``(x, y, z)`` = ``(cW, cH, cD)`` convention — see
        ``commit_block()`` for the conversion.
    cache_grid : tuple[int, int, int]
        ``(grid_side, grid_side, grid_side)`` — the slot count per
        axis as a tuple.  Retained for completeness; in practice
        ``grid_side`` is read directly.
    """

    grid_side: int
    block_size: int
    overlap: int

    @property
    def padded_block_size(self) -> int:
        """Actual slot side length in voxels, including the overlap border."""
        return self.block_size + 2 * self.overlap

    @property
    def n_slots(self) -> int:
        """Total number of slots in the cache, including the reserved slot 0."""
        return self.grid_side**3

    @property
    def cache_shape(self) -> tuple[int, int, int]:
        """Numpy-order ``(cD, cH, cW)`` shape of the cache texture."""
        s = self.grid_side * self.padded_block_size
        return (s, s, s)

    @property
    def cache_grid(self) -> tuple[int, int, int]:
        """Slot count per axis as a tuple ``(grid_side, grid_side, grid_side)``."""
        return (self.grid_side, self.grid_side, self.grid_side)


def compute_block_cache_parameters_3d(
    block_size: int,
    gpu_budget_bytes: int,
    overlap: int = 1,
    bytes_per_voxel: int = 4,
) -> BlockCacheParameters3D:
    """Compute cache dimensions that fit within the GPU memory budget.

    Parameters
    ----------
    block_size : int
        Logical brick side length in voxels.
    gpu_budget_bytes : int
        Maximum byte budget for the cache texture.
    overlap : int
        Border voxels duplicated on each side of every slot.
    bytes_per_voxel : int
        Bytes per voxel (4 for float32).

    Returns
    -------
    params : BlockCacheParameters3D
        Cache sizing metadata.
    """
    padded = block_size + 2 * overlap
    bytes_per_brick = padded**3 * bytes_per_voxel
    max_slots = gpu_budget_bytes // bytes_per_brick
    grid_side = int(max_slots ** (1.0 / 3.0))
    grid_side = max(grid_side, 2)
    return BlockCacheParameters3D(
        grid_side=grid_side, block_size=block_size, overlap=overlap
    )


def build_cache_texture_3d(
    params: BlockCacheParameters3D,
) -> tuple[np.ndarray, gfx.Texture]:
    """Allocate the fixed-size cache texture, zeroed.

    Parameters
    ----------
    params : BlockCacheParameters3D
        Cache sizing metadata.

    Returns
    -------
    cache_data : np.ndarray
        CPU-side backing array of shape ``(cD, cH, cW)``, dtype float32.
    cache_tex : gfx.Texture
        pygfx 3-D texture wrapping ``cache_data``.
    """
    cache_data = np.zeros(params.cache_shape, dtype=np.float32)
    cache_tex = gfx.Texture(cache_data, dim=3)
    return cache_data, cache_tex


def commit_block_3d(
    cache_data: np.ndarray,
    cache_tex: gfx.Texture,
    grid_pos: tuple[int, int, int],
    padded_block_size: int,
    data: np.ndarray,
) -> None:
    """Write a padded brick into the CPU cache array and schedule a GPU upload.

    The GPU transfer is deferred until the next ``renderer.render()`` call
    (pygfx ``update_range`` semantics).

    Parameters
    ----------
    cache_data : np.ndarray
        CPU-side backing array for the cache texture.
    cache_tex : gfx.Texture
        pygfx texture to mark dirty for upload.
    grid_pos : tuple[int, int, int]
        Slot grid indices (slot_index_z, slot_index_y, slot_index_x)
        in numpy axis order.
    padded_block_size : int
        Side length of the padded brick in voxels
        (block_size + 2 * overlap).
    data : np.ndarray
        Float32 block of data to commit of shape
        (padded_block_size, padded_block_size, padded_block_size).
    """
    slot_index_z, slot_index_y, slot_index_x = grid_pos
    z_start = slot_index_z * padded_block_size
    y_start = slot_index_y * padded_block_size
    x_start = slot_index_x * padded_block_size
    cache_data[
        z_start : z_start + padded_block_size,
        y_start : y_start + padded_block_size,
        x_start : x_start + padded_block_size,
    ] = data
    cache_tex.update_range(
        offset=(x_start, y_start, z_start),
        size=(padded_block_size, padded_block_size, padded_block_size),
    )
