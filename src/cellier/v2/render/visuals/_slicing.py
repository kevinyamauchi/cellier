# src/cellier/v2/render/visuals/_slicing.py
"""Shared slice-index to voxel-index conversion for render-layer visuals.

All image and label visuals -- in-memory and multiscale, single- and
multi-channel, 2D and 3D -- snap a world-space slice position to an integer
voxel index using a single rule defined here.  Keeping it in one place
guarantees that the same physical world position selects the same data plane
regardless of which visual family renders it.
"""

from __future__ import annotations

import numpy as np


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
