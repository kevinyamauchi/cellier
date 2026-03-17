"""Volume material using brick-cache + LUT indirection rendering."""

import pygfx as gfx
from pygfx.resources import Buffer


class VolumeBlockMaterial(gfx.VolumeBasicMaterial):
    """Volume material using brick-cache + LUT indirection rendering.

    Inherits from ``VolumeBasicMaterial`` (not bare ``Material``) to get
    the standard ``clim``, ``gamma``, ``map``, and ``interpolation``
    properties, as well as the required ``clipping_planes`` uniform
    field (see lesson L1 in the implementation plan).

    Parameters
    ----------
    lut_texture : gfx.Texture
        RGBA8UI 3D texture — the per-brick address lookup table.
        Shape ``(gD, gH, gW, 4)``.
        Channels: ``(tile_x, tile_y, tile_z, level)``.
    lut_params_buffer : Buffer
        Uniform buffer containing ``block_size``, ``cache_size``,
        ``lut_size``, ``lut_offset``.
        Must use ``LUT_PARAMS_DTYPE`` from ``uniforms.py``.
    clim : tuple[float, float]
        Contrast limits applied before colourmap lookup.
    map : gfx.TextureMap, optional
        1D colourmap texture. If ``None`` a greyscale ramp is used.
    interpolation : str
        Sampler filter for the cache texture. ``"linear"`` or
        ``"nearest"``.
    """

    def __init__(
        self,
        lut_texture: gfx.Texture,
        lut_params_buffer: Buffer,
        clim: tuple[float, float] = (0.0, 1.0),
        map: gfx.TextureMap | None = None,
        interpolation: str = "linear",
        **kwargs,
    ) -> None:
        super().__init__(clim=clim, map=map, interpolation=interpolation, **kwargs)
        self.lut_texture = lut_texture
        self.lut_params_buffer = lut_params_buffer
