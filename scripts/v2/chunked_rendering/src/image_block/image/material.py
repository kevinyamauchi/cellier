"""Image material using tile-cache + LUT indirection rendering.

Inherits from ``ImageBasicMaterial`` to get ``clim``, ``gamma``,
``map``, and ``interpolation`` properties.  Carries extra texture and
buffer references for the tile cache system.
"""

from __future__ import annotations

import pygfx as gfx
from pygfx.resources import Buffer


class ImageBlockMaterial(gfx.ImageBasicMaterial):
    """Image material using tile-cache + LUT indirection rendering.

    Parameters
    ----------
    cache_texture : gfx.Texture
        2D float32 texture -- the fixed-size tile cache.
    lut_texture : gfx.Texture
        RGBA float32 2D texture -- the per-tile address lookup table.
        Shape ``(gH, gW, 4)``.
    lut_params_buffer : Buffer
        Uniform buffer containing LUT spatial parameters.
    block_scales_buffer : Buffer
        Uniform buffer with per-level scale factors (10 x vec4).
    clim : tuple[float, float]
        Contrast limits.
    map : gfx.TextureMap, optional
        1D colourmap texture.
    interpolation : str
        Sampler filter for the cache texture.
    """

    def __init__(
        self,
        cache_texture: gfx.Texture,
        lut_texture: gfx.Texture,
        lut_params_buffer: Buffer,
        block_scales_buffer: Buffer,
        clim: tuple[float, float] = (0.0, 1.0),
        map: gfx.TextureMap | None = None,
        interpolation: str = "linear",
        **kwargs,
    ) -> None:
        super().__init__(
            clim=clim,
            map=map,
            interpolation=interpolation,
            **kwargs,
        )
        self.cache_texture = cache_texture
        self.lut_texture = lut_texture
        self.lut_params_buffer = lut_params_buffer
        self.block_scales_buffer = block_scales_buffer
