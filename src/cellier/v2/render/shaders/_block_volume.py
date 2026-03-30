"""LUT brick-cache volume shader: material, shader class, and uniform buffers.

Importing this module registers ``VolumeBlockShader`` as the pygfx render
function for ``(Volume, VolumeBlockMaterial)`` pairs via the
``@register_wgpu_render_function`` decorator.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx
import wgpu
from pygfx.objects import Volume
from pygfx.renderers.wgpu import (
    Binding,
    GfxSampler,
    GfxTextureView,
    register_wgpu_render_function,
)
from pygfx.renderers.wgpu.shaders.volumeshader import BaseVolumeShader
from pygfx.resources import Buffer

if TYPE_CHECKING:
    from cellier.v2.render.block_cache import BlockCacheParameters3D
    from cellier.v2.render.lut_indirection import BlockLayout3D

_WGSL_PATH = Path(__file__).parent / "wgsl" / "volume_block.wgsl"
_VOLUME_BLOCK_WGSL = _WGSL_PATH.read_text()

_vertex_and_fragment = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------


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
        ``lut_size``, ``vol_size``, and ``overlap``.
    block_scales_buffer : Buffer
        Uniform buffer with per-level scale factors (10 x vec4).
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


# ---------------------------------------------------------------------------
# Shader
# ---------------------------------------------------------------------------


@register_wgpu_render_function(Volume, VolumeBlockMaterial)
class VolumeBlockShader(BaseVolumeShader):
    """Shader for LUT-based brick-cache volume rendering."""

    type = "render"

    def get_bindings(self, wobject, shared, scene):
        geometry = wobject.geometry
        material = wobject.material

        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        # Proxy texture — defines the volume bounding box.
        proxy_view = GfxTextureView(geometry.grid)
        proxy_sampler = GfxSampler(material.interpolation, "clamp")
        bindings.append(
            Binding("s_img", "sampler/filtering", proxy_sampler, "FRAGMENT")
        )
        bindings.append(
            Binding("t_img", "texture/auto", proxy_view, _vertex_and_fragment)
        )

        # Cache texture + sampler.
        cache_view = GfxTextureView(material.cache_texture)
        cache_sampler = GfxSampler(material.interpolation, "clamp")
        bindings.append(
            Binding("s_cache", "sampler/filtering", cache_sampler, "FRAGMENT")
        )
        bindings.append(Binding("t_cache", "texture/auto", cache_view, "FRAGMENT"))

        # Colourmap.
        if material.map is not None:
            bindings.extend(self.define_img_colormap(material.map))

        # LUT texture.
        lut_view = GfxTextureView(material.lut_texture)
        bindings.append(Binding("t_lut", "texture/auto", lut_view, "FRAGMENT"))

        # Uniform buffers.
        bindings.append(
            Binding(
                "u_lut_params",
                "buffer/uniform",
                material.lut_params_buffer,
                "FRAGMENT",
                structname="LutParams",
            )
        )
        bindings.append(
            Binding(
                "u_block_scales",
                "buffer/uniform",
                material.block_scales_buffer,
                "FRAGMENT",
                structname="BlockScales",
            )
        )

        bindings = dict(enumerate(bindings))
        self.define_bindings(0, bindings)
        return {0: bindings}

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.front,
        }

    def get_render_info(self, wobject, shared):
        return {"indices": (36, 1)}

    def get_code(self):
        return _VOLUME_BLOCK_WGSL


# ---------------------------------------------------------------------------
# Uniform buffer builders
# ---------------------------------------------------------------------------

LUT_PARAMS_DTYPE = np.dtype(
    [
        ("block_size_x", "<f4"),
        ("block_size_y", "<f4"),
        ("block_size_z", "<f4"),
        ("overlap", "<f4"),
        ("cache_size_x", "<f4"),
        ("cache_size_y", "<f4"),
        ("cache_size_z", "<f4"),
        ("_pad1", "<f4"),
        ("lut_size_x", "<f4"),
        ("lut_size_y", "<f4"),
        ("lut_size_z", "<f4"),
        ("_pad2", "<f4"),
        ("vol_size_x", "<f4"),
        ("vol_size_y", "<f4"),
        ("vol_size_z", "<f4"),
        ("_pad3", "<f4"),
    ]
)


def build_lut_params_buffer_3d(
    base_layout: BlockLayout3D,
    cache_info: BlockCacheParameters3D,
    proxy_voxels_per_brick: float | None = None,
) -> Buffer:
    """Build the LUT parameter uniform buffer.

    Parameters
    ----------
    base_layout : BlockLayout3D
        Layout of the finest (level 1) resolution.
    cache_info : BlockCacheParameters3D
        Cache sizing metadata.
    proxy_voxels_per_brick : float or None
        When the proxy texture uses grid dims (1 texel per brick), pass
        ``block_size`` here so all spatial uniforms are normalised to
        grid units.  ``vol_size_*`` always carries the real padded voxel
        extent for step-density calibration in the fragment shader.

    Returns
    -------
    Buffer
        Uniform buffer bound as ``u_lut_params`` in the shader.
    """
    gd, gh, gw = base_layout.grid_dims
    pd, ph, pw = base_layout.padded_shape
    bs = float(base_layout.block_size)
    ov = float(cache_info.overlap)

    data = np.zeros((), dtype=LUT_PARAMS_DTYPE)

    if proxy_voxels_per_brick is not None:
        p = float(proxy_voxels_per_brick)
        ov_u = ov / p
        padded_u = 1.0 + 2.0 * ov_u
        gs = float(cache_info.grid_side)
        cs_u = gs * padded_u

        data["block_size_x"] = 1.0
        data["block_size_y"] = 1.0
        data["block_size_z"] = 1.0
        data["overlap"] = ov_u
        data["cache_size_x"] = cs_u
        data["cache_size_y"] = cs_u
        data["cache_size_z"] = cs_u
    else:
        cd, ch, cw = cache_info.cache_shape
        data["block_size_x"] = bs
        data["block_size_y"] = bs
        data["block_size_z"] = bs
        data["overlap"] = ov
        data["cache_size_x"] = float(cw)
        data["cache_size_y"] = float(ch)
        data["cache_size_z"] = float(cd)

    data["lut_size_x"] = float(gw)
    data["lut_size_y"] = float(gh)
    data["lut_size_z"] = float(gd)
    data["vol_size_x"] = float(pw)
    data["vol_size_y"] = float(ph)
    data["vol_size_z"] = float(pd)

    return Buffer(data, force_contiguous=True)


MAX_LEVELS = 10

BLOCK_SCALES_DTYPE = np.dtype([(f"scale_{i}", "<f4", (4,)) for i in range(MAX_LEVELS)])


def build_block_scales_buffer_3d(
    level_scale_vecs_data: list[np.ndarray],
) -> Buffer:
    """Build the block-scales uniform buffer.

    Level 0 is reserved (all zeros — renders black/out-of-bounds).
    Level k (1-indexed) gets ``1 / scale_factor`` per axis.

    Input vectors are in data-axis order ``(sz, sy, sx)``; shader
    fields ``[0], [1], [2]`` are ``(x=W, y=H, z=D)`` so the
    assignment reverses the index.

    Parameters
    ----------
    level_scale_vecs_data : list[np.ndarray]
        Per-level scale vectors in data order, e.g.
        ``[array([1,1,1]), array([2,2,2]), array([4,4,4])]``.

    Returns
    -------
    Buffer
        Uniform buffer bound as ``u_block_scales`` in the shader.
    """
    data = np.zeros((), dtype=BLOCK_SCALES_DTYPE)

    for k in range(1, min(len(level_scale_vecs_data) + 1, MAX_LEVELS)):
        sv = level_scale_vecs_data[k - 1]  # data order: [sz, sy, sx]
        # shader x = W = data axis 2 (sx)
        data[f"scale_{k}"][0] = 1.0 / float(sv[2])
        # shader y = H = data axis 1 (sy)
        data[f"scale_{k}"][1] = 1.0 / float(sv[1])
        # shader z = D = data axis 0 (sz)
        data[f"scale_{k}"][2] = 1.0 / float(sv[0])

    return Buffer(data, force_contiguous=True)
