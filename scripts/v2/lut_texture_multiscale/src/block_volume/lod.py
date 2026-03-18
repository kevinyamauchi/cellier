"""LOD level selection for bricked volume rendering.

Phase 1: distance-based selection.  Each brick in the base grid is
assigned a level depending on how far its centre is from the camera.
"""

from __future__ import annotations

import numpy as np

from block_volume.layout import BlockLayout
from block_volume.tile_manager import BrickKey


def select_levels(
    base_layout: BlockLayout,
    n_levels: int,
    camera_pos: np.ndarray,
    thresholds: list[float] | None = None,
) -> dict[BrickKey, int]:
    """Assign a desired LOD level to every brick in the base grid.

    Parameters
    ----------
    base_layout : BlockLayout
        Layout of the finest (level 1) resolution.
    n_levels : int
        Total number of LOD levels available (e.g. 3).
    camera_pos : np.ndarray
        Camera position in world space ``(x, y, z)``.
    thresholds : list[float] or None
        Distance cutoffs for level transitions.  ``thresholds[i]`` is
        the distance beyond which level ``i + 2`` is selected instead
        of ``i + 1``.  Length must be ``n_levels - 1``.  If ``None``,
        defaults are computed from the volume diagonal.

    Returns
    -------
    required : dict[BrickKey, int]
        ``{BrickKey(level, gz, gy, gx): level}`` for every brick.
    """
    gd, gh, gw = base_layout.grid_dims
    bs = base_layout.block_size

    if thresholds is None:
        # Default: level 1 within 1× diagonal, level 2 within 2×, etc.
        diag = np.sqrt(sum(s**2 for s in base_layout.volume_shape))
        thresholds = [diag * (i + 1) for i in range(n_levels - 1)]

    required: dict[BrickKey, int] = {}

    for gz in range(gd):
        for gy in range(gh):
            for gx in range(gw):
                # Brick centre in world space (x=W, y=H, z=D).
                cx = (gx + 0.5) * bs
                cy = (gy + 0.5) * bs
                cz = (gz + 0.5) * bs
                centre = np.array([cx, cy, cz], dtype=np.float64)
                dist = np.linalg.norm(centre - camera_pos)

                # Determine level.
                level = 1
                for k, thr in enumerate(thresholds):
                    if dist > thr:
                        level = k + 2
                    else:
                        break
                level = min(level, n_levels)

                # Convert base-grid coords to this level's grid coords.
                scale = 2 ** (level - 1)
                key = BrickKey(
                    level=level,
                    gz=gz // scale,
                    gy=gy // scale,
                    gx=gx // scale,
                )
                # Multiple base bricks may map to the same coarse key;
                # dict dedup is intentional.
                required[key] = level

    return required
