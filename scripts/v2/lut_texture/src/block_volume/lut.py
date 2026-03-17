"""LUT (lookup texture) construction for bricked volume rendering.

The LUT is an RGBA8UI 3D texture where each voxel encodes
(tile_x, tile_y, tile_z, level) as uint8 values. For single-LOD
rendering, the LUT is an identity mapping where each brick maps
to itself.

Notes
-----
Channel assignments follow the axis convention:

- ``lut[d, h, w, 0]`` = tile_x = w (cache W axis)
- ``lut[d, h, w, 1]`` = tile_y = h (cache H axis)
- ``lut[d, h, w, 2]`` = tile_z = d (cache D axis)
- ``lut[d, h, w, 3]`` = level  = 1 (base level; 0 is reserved)
"""

import numpy as np
import pygfx as gfx

from block_volume.layout import BlockLayout


def build_identity_lut(layout: BlockLayout) -> gfx.Texture:
    """Build the identity lookup texture for single-LOD rendering.

    Each brick maps to itself. The four RGBA channels encode
    ``(tile_x, tile_y, tile_z, level)`` as uint8.

    Parameters
    ----------
    layout : BlockLayout
        Brick layout parameters.

    Returns
    -------
    texture : gfx.Texture
        RGBA8UI 3D texture of shape ``(gD, gH, gW, 4)``.
        Declared as ``texture_3d<u32>`` in WGSL.
    """
    gd, gh, gw = layout.grid_dims

    lut = np.zeros((gd, gh, gw, 4), dtype=np.uint8)

    d_idx, h_idx, w_idx = np.meshgrid(
        np.arange(gd, dtype=np.uint8),
        np.arange(gh, dtype=np.uint8),
        np.arange(gw, dtype=np.uint8),
        indexing="ij",
    )
    lut[..., 0] = w_idx
    lut[..., 1] = h_idx
    lut[..., 2] = d_idx
    lut[..., 3] = 1

    # NOTE (L3): format="rgba8uint" is required so pygfx declares
    # texture_3d<u32> in WGSL. Without it, uint8 data maps to
    # rgba8unorm (texture_3d<f32>), which breaks textureLoad() usage.
    return gfx.Texture(lut, dim=3, format="rgba8uint")
