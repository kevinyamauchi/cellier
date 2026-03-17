"""Cache texture construction for bricked volume rendering.

Pads the volume to the layout's padded shape and wraps it in a pygfx
Texture. In phase 0 (single LOD), the cache IS the padded volume with
no brick rearrangement.

Notes
-----
Axis order for ``gfx.Texture`` constructor matches numpy: (D, H, W).
However ``texture.update_range(offset, size)`` uses **(W, H, D)** order.
"""

import numpy as np
import pygfx as gfx

from block_volume.layout import BlockLayout


def build_cache_texture(volume: np.ndarray, layout: BlockLayout) -> gfx.Texture:
    """Build the GPU cache texture from a float32 volume array.

    Pads the volume to ``layout.padded_shape`` with zeros and returns a
    ``gfx.Texture`` backed by the padded array.

    Parameters
    ----------
    volume : np.ndarray
        Float32 array of shape (D, H, W). Values should be in [0, 1].
    layout : BlockLayout
        Brick layout parameters for this volume.

    Returns
    -------
    texture : gfx.Texture
        A 3D texture suitable for use as ``geometry.grid``.
    """
    d, h, w = volume.shape

    padded = np.zeros(layout.padded_shape, dtype=np.float32)
    padded[:d, :h, :w] = volume.astype(np.float32)

    return gfx.Texture(padded, dim=3)
