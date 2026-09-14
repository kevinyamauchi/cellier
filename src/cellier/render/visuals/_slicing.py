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

from cellier._rounding import round_half_up_clamped

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cellier.transform import AxisAlignedBoundingBox


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

    The arithmetic itself lives in
    :func:`cellier._rounding.round_half_up_clamped`, shared with the
    non-uniform axis transform's nearest-neighbour lookup so the two cannot
    drift apart.  This function keeps its name, its voxel-specific meaning
    and every call site.

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
    return round_half_up_clamped(raw, size)


def axis_selections_from_box(
    box: AxisAlignedBoundingBox,
    level_shape: tuple[int, ...],
    windows: Mapping[int, tuple[int, int]] | None = None,
) -> tuple[int | tuple[int, int], ...]:
    """Turn a voxel-space bounding box into a datastore's per-axis selection.

    The one assembler for every image and label family, replacing the
    per-family ``_build_axis_selections_*``.  Its input is a box already
    pulled back into level-k voxel space, and design 3.7 step 4 is its three
    branches, per data axis:

    * **unbounded** -- the region does not constrain this axis, so the whole
      extent is fetched.  This is what a displayed axis looks like until the
      viewport crop lands, and it reproduces today's ``(0, shape[axis])``
      exactly.
    * **collapsed** (``lo == hi``) -- one plane.  The position is snapped with
      :func:`round_world_to_voxel`, the same half-up rule and the same
      function as before this migration, and the axis is dropped from the
      returned array by ``get_data``.
    * **a real slab** -- the third branch, reached when a thickness is asked
      for.  The bounds are widened to whole voxels: any voxel the slab touches
      is fetched.

    An axis that comes back **half** bounded on a collapsed axis is a shear:
    the region's preimage is a slanted hyperplane with no single voxel index
    on it, and there is no honest answer.  See D7.

    Parameters
    ----------
    box : AxisAlignedBoundingBox
        The region's bounds in level-k voxel space, one entry per data axis.
    level_shape : tuple[int, ...]
        Extent of the data at this level, for clamping.
    windows : Mapping[int, tuple[int, int]] or None
        Per-axis ``(start, stop)`` overrides for axes whose window comes from
        somewhere other than the region -- a multiscale brick's padded grid
        window, say.  These win over the box.

    Returns
    -------
    tuple[int | tuple[int, int], ...]
        One entry per data axis, in ascending data-axis order.

    Raises
    ------
    ValueError
        If an axis is bounded on one side only.
    """
    windows = dict(windows or {})
    result: list[int | tuple[int, int]] = []
    for axis, size in enumerate(level_shape):
        if axis in windows:
            result.append(tuple(int(value) for value in windows[axis]))
            continue
        low = float(box.min_coordinate[axis])
        high = float(box.max_coordinate[axis])
        low_unbounded = np.isneginf(low)
        high_unbounded = np.isposinf(high)
        if low_unbounded and high_unbounded:
            result.append((0, int(size)))
        elif low_unbounded or high_unbounded:
            raise ValueError(
                f"Axis {axis} of this region is bounded on one side only "
                f"([{low}, {high}]).  That happens when the data -> world "
                f"transform has a cross-term on a collapsed axis: the "
                f"preimage of the slice plane is a slanted hyperplane and no "
                f"single voxel index lies on it.  Remove the shear -- "
                f"_check_transform_no_rotation imposes the same restriction "
                f"on the multiscale brick shader -- or wait for the oblique "
                f"path."
            )
        elif low == high:
            result.append(round_world_to_voxel(low, int(size)))
        else:
            # Both ends are clamped into [0, size], so a slab that misses the
            # data entirely comes back as an in-range empty window rather than
            # an out-of-range or inverted one.
            start = min(max(0, int(np.floor(low + 0.5))), int(size))
            stop = min(int(size), int(np.floor(high + 0.5)) + 1)
            result.append((start, max(start, stop)))
    return tuple(result)
