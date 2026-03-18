"""Shader class for LUT-based brick-cache volume rendering.

Phase 1: binds two 3D textures:
- ``t_img`` (via ``geometry.grid``): a proxy texture whose dimensions
  define the volume bounding box.
- ``t_cache`` + ``s_cache``: the actual fixed-size brick cache texture
  used for sampling.

Registers ``VolumeBlockShader`` as the render function for
``(Volume, VolumeBlockMaterial)`` pairs in pygfx's wgpu renderer.
"""

from pathlib import Path

import wgpu
from pygfx.objects import Volume
from pygfx.renderers.wgpu import (
    Binding,
    GfxSampler,
    GfxTextureView,
    register_wgpu_render_function,
)
from pygfx.renderers.wgpu.shaders.volumeshader import BaseVolumeShader

from block_volume.material import VolumeBlockMaterial

_WGSL_DIR = Path(__file__).parent / "wgsl"
VOLUME_BLOCK_WGSL = (_WGSL_DIR / "volume_block.wgsl").read_text()

_vertex_and_fragment = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT


@register_wgpu_render_function(Volume, VolumeBlockMaterial)
class VolumeBlockShader(BaseVolumeShader):
    """Shader for LUT-based brick-cache volume rendering."""

    type = "render"

    def get_bindings(self, wobject, shared, scene):
        """Return all GPU resource bindings for the volume block shader."""
        geometry = wobject.geometry
        material = wobject.material

        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        # Proxy texture (volume-sized) — used by get_vol_geometry() for the
        # bounding box and by fs_main for sizef = textureDimensions(t_img).
        # s_img is needed because volume_common.wgsl's sample_vol() references it
        # even though we use sample_vol_lut() instead.
        proxy_view = GfxTextureView(geometry.grid)
        proxy_sampler = GfxSampler(material.interpolation, "clamp")
        bindings.append(
            Binding("s_img", "sampler/filtering", proxy_sampler, "FRAGMENT")
        )
        bindings.append(
            Binding("t_img", "texture/auto", proxy_view, _vertex_and_fragment)
        )

        # Cache texture + sampler — used by sample_vol_lut() for actual data.
        cache_view = GfxTextureView(material.cache_texture)
        cache_sampler = GfxSampler(material.interpolation, "clamp")
        bindings.append(
            Binding("s_cache", "sampler/filtering", cache_sampler, "FRAGMENT")
        )
        bindings.append(
            Binding("t_cache", "texture/auto", cache_view, "FRAGMENT")
        )

        # Colourmap.
        if material.map is not None:
            bindings.extend(self.define_img_colormap(material.map))

        # LUT texture.
        lut_view = GfxTextureView(material.lut_texture)
        bindings.append(
            Binding("t_lut", "texture/auto", lut_view, "FRAGMENT")
        )

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

        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)
        return {0: bindings}

    def get_pipeline_info(self, wobject, shared):
        """Return pipeline configuration."""
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.front,
        }

    def get_render_info(self, wobject, shared):
        """Return draw-call parameters."""
        return {"indices": (36, 1)}

    def get_code(self):
        """Return the WGSL source for this shader."""
        return VOLUME_BLOCK_WGSL
