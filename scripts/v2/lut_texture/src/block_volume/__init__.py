"""LUT-based brick-cache volume renderer for pygfx.

Phase 0: single-LOD renderer where the entire volume fits in GPU
memory. The LUT is an identity mapping, but all rendering machinery
(LUT texture, uniform buffer, custom WGSL shader) is fully exercised.

Importing this module registers the ``VolumeBlockShader`` with pygfx's
wgpu renderer for ``(Volume, VolumeBlockMaterial)`` pairs.
"""

import numpy as np
import pygfx as gfx

from block_volume.cache import build_cache_texture
from block_volume.layout import BLOCK_SIZE_DEFAULT, BlockLayout
from block_volume.lut import build_identity_lut
from block_volume.material import VolumeBlockMaterial
from block_volume.uniforms import build_lut_params_buffer

# Importing shader registers the render function with pygfx.
import block_volume.shader as _shader  # noqa: F401

__all__ = [
    "BlockLayout",
    "VolumeBlockMaterial",
    "make_block_volume",
]


def make_block_volume(
    volume: np.ndarray,
    block_size: int = BLOCK_SIZE_DEFAULT,
    colormap: gfx.TextureMap | None = None,
    clim: tuple[float, float] = (0.0, 1.0),
) -> gfx.Volume:
    """Create a pygfx Volume using LUT-based brick-cache rendering.

    Parameters
    ----------
    volume : np.ndarray
        Float32 array of shape ``(D, H, W)``. Values should be
        normalised to [0, 1].
    block_size : int
        Brick side length in voxels. Must satisfy
        ``max(grid_dims) <= 255``.
    colormap : gfx.TextureMap, optional
        Colourmap for intensity mapping. Defaults to greyscale.
    clim : tuple[float, float]
        Contrast limits.

    Returns
    -------
    vol : gfx.Volume
        A pygfx Volume that can be added directly to a scene.
    """
    layout = BlockLayout.from_volume_shape(volume.shape, block_size=block_size)

    cache_tex = build_cache_texture(volume, layout)
    lut_tex = build_identity_lut(layout)
    lut_params = build_lut_params_buffer(layout)

    if colormap is None:
        colormap = gfx.cm.gray

    material = VolumeBlockMaterial(
        lut_texture=lut_tex,
        lut_params_buffer=lut_params,
        clim=clim,
        map=colormap,
    )

    geometry = gfx.Geometry(grid=cache_tex)
    return gfx.Volume(geometry, material)
