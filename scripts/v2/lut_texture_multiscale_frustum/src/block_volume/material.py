"""Volume material using brick-cache + LUT indirection rendering.

Phase 1: inherits from ``VolumeIsoMaterial`` to get ``threshold``,
``shininess``, ``step_size``, ``substep_size`` uniforms for
isosurface raycasting.  Carries a separate ``cache_texture`` so the
geometry proxy texture can define the bounding box independently.
"""

import pygfx as gfx
from pygfx.resources import Buffer


class VolumeBlockMaterial(gfx.VolumeIsoMaterial):
    """Volume material using brick-cache + LUT indirection rendering.

    Parameters
    ----------
    cache_texture : gfx.Texture
        3D float32 texture — the fixed-size brick cache.
    lut_texture : gfx.Texture
        RGBA8UI 3D texture — the per-brick address lookup table.
    lut_params_buffer : Buffer
        Uniform buffer with ``block_size``, ``cache_size``,
        ``lut_size``, ``lut_offset``.
    block_scales_buffer : Buffer
        Uniform buffer with per-level scale factors (10 × vec4).
    clim : tuple[float, float]
        Contrast limits.
    map : gfx.TextureMap, optional
        1D colourmap texture.
    interpolation : str
        Sampler filter for the cache texture.
    threshold : float
        Isosurface threshold.
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
        threshold: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(
            clim=clim,
            map=map,
            interpolation=interpolation,
            threshold=threshold,
            **kwargs,
        )
        self.cache_texture = cache_texture
        self.lut_texture = lut_texture
        self.lut_params_buffer = lut_params_buffer
        self.block_scales_buffer = block_scales_buffer
