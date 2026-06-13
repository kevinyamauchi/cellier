# src/cellier/v2/render/visuals/_slicing.py
"""Shared slice-index to voxel-index conversion for render-layer visuals.

All image and label visuals -- in-memory and multiscale, single- and
multi-channel, 2D and 3D -- snap a world-space slice position to an integer
voxel index using a single rule defined here.  Keeping it in one place
guarantees that the same physical world position selects the same data plane
regardless of which visual family renders it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cellier.transform import AffineTransform


def round_world_to_voxel(raw: float, size: int) -> int:
    """Snap a level-k voxel-space position to the nearest voxel index.

    Uses round-half-up: ``floor(raw + 0.5)``.  This is the unique rounding
    rule consistent with the center-at-integer convention used everywhere
    else in the render layer -- voxel ``i`` is centered at integer ``i`` and
    spans ``[i - 0.5, i + 0.5)``, the geometry extent is ``[-0.5, N - 0.5]``,
    and the shader samples at ``pos = local + 0.5``.  Picking the nearest
    voxel center (rather than ``floor(raw)``, which would adopt a
    corner-at-integer convention) keeps the sliced axes on the same grid as
    the displayed axes.

    Ties (exact half-integer positions) round toward ``+inf``, which is
    deterministic and monotonic for slider scrubbing.  The result is clamped
    to the valid index range ``[0, size - 1]``.

    Parameters
    ----------
    raw : float
        Slice position already mapped into level-k voxel space.
    size : int
        Extent of the axis at this level (``store_shape[axis]`` for in-memory
        data, ``level_shape[axis]`` for multiscale data).  Used for clamping.

    Returns
    -------
    int
        Voxel index in ``[0, size - 1]``.
    """
    idx = int(np.floor(raw + 0.5))
    return max(0, min(idx, size - 1))


def map_world_slice_to_voxel(
    slice_indices: dict[int, int],
    ndim: int,
    world_to_voxel: AffineTransform,
    level_shape: tuple[int, ...],
) -> dict[int, int]:
    """Map world-space slice positions to rounded level-k voxel indices.

    This is the single world->voxel mapping used by every image and label
    visual -- in-memory and multiscale alike.  Each family supplies a
    *forward* transform whose ``map_coordinates`` takes a world-space point to
    level-k voxel space:

    - In-memory visuals pass ``AffineTransform(matrix=transform.inverse_matrix)``
      (the inverse of the data->world transform); there is a single level so
      ``level_shape`` is the store shape.
    - Multiscale visuals pass the precomposed ``world_to_level_k`` transform
      (``inv_level_k @ inv_visual``) and the level-k shape.

    Both express the same operation (world -> voxel); keeping it in one place
    guarantees the same physical world position selects the same data plane
    regardless of which visual family renders it.  Each sliced axis is snapped
    with :func:`round_world_to_voxel` (round-half-up) and clamped to a valid
    index.

    Only the axes present in ``slice_indices`` (the non-displayed axes) are
    mapped and returned; displayed axes are handled by the caller's
    axis-selection assembly.

    Parameters
    ----------
    slice_indices : dict[int, int]
        Axis -> world-space slice position, for the sliced (non-displayed) axes.
    ndim : int
        Number of data dimensions (the dimensionality of ``world_to_voxel``).
    world_to_voxel : AffineTransform
        Forward transform mapping a world-space point to level-k voxel space.
    level_shape : tuple[int, ...]
        Shape of the data at this level, used for clamping each axis.

    Returns
    -------
    dict[int, int]
        Axis -> level-k voxel index, for the same axes as ``slice_indices``.
    """
    if not slice_indices:
        return {}

    world_pt = np.zeros(ndim, dtype=np.float64)
    for axis, world_pos in slice_indices.items():
        world_pt[axis] = float(world_pos)

    voxel_pt = world_to_voxel.map_coordinates(world_pt.reshape(1, -1)).flatten()

    return {
        axis: round_world_to_voxel(float(voxel_pt[axis]), level_shape[axis])
        for axis in slice_indices
    }
