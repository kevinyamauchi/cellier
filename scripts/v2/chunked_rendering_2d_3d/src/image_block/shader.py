"""Shader class for LUT-based tile-cache 2D image rendering.

Registers ``ImageBlockShader`` as the render function for
``(Image, ImageBlockMaterial)`` pairs in pygfx's wgpu renderer.

The binding layout and ``get_code`` override pattern were validated
in ``test_image_block_shader.py`` prior to this implementation.
"""

from __future__ import annotations

from pathlib import Path

import wgpu
from pygfx.objects import Image
from pygfx.renderers.wgpu import (
    Binding,
    GfxSampler,
    GfxTextureView,
    register_wgpu_render_function,
)
from pygfx.renderers.wgpu.shaders.imageshader import ImageShader

from image_block.material import ImageBlockMaterial

_WGSL_DIR = Path(__file__).parent / "wgsl"
IMAGE_BLOCK_WGSL = (_WGSL_DIR / "image_block.wgsl").read_text()

_vertex_and_fragment = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT


@register_wgpu_render_function(Image, ImageBlockMaterial)
class ImageBlockShader(ImageShader):
    """Shader for LUT-based tile-cache 2D image rendering."""

    type = "render"

    def get_bindings(self, wobject, shared, scene):
        """Return all GPU resource bindings for the image block shader.

        Binds:
        - t_img / s_img: proxy texture (grid-dim), drives geometry
        - t_cache / s_cache: tile cache texture
        - t_lut: float32 LUT texture (textureLoad, no sampler)
        - u_lut_params: LutParams uniform
        - u_block_scales: BlockScales uniform
        """
        geometry = wobject.geometry
        material = wobject.material

        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        # Proxy texture -- grid-dim, drives get_im_geometry() for quad size.
        proxy_view = GfxTextureView(geometry.grid)
        proxy_sampler = GfxSampler(material.interpolation, "clamp")
        bindings.append(
            Binding("s_img", "sampler/filtering", proxy_sampler, "FRAGMENT")
        )
        bindings.append(
            Binding(
                "t_img", "texture/auto", proxy_view, _vertex_and_fragment
            )
        )

        # Cache texture + sampler -- actual tile data.
        cache_view = GfxTextureView(material.cache_texture)
        cache_sampler = GfxSampler(material.interpolation, "clamp")
        bindings.append(
            Binding(
                "s_cache", "sampler/filtering", cache_sampler, "FRAGMENT"
            )
        )
        bindings.append(
            Binding("t_cache", "texture/auto", cache_view, "FRAGMENT")
        )

        # Colourmap.
        if material.map is not None:
            bindings.extend(self.define_img_colormap(material.map))

        # LUT texture (textureLoad, no sampler needed).
        lut_view = GfxTextureView(material.lut_texture)
        bindings.append(
            Binding("t_lut", "texture/auto", lut_view, "FRAGMENT")
        )

        # Uniform buffers.
        # structname= tells pygfx to auto-generate the WGSL struct
        # from the numpy dtype. Do NOT write the struct manually in WGSL.
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
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        """Return draw-call parameters."""
        return {"indices": (6, 1)}

    def get_code(self):
        """Return the WGSL source for this shader."""
        return IMAGE_BLOCK_WGSL
