"""Shader class for LUT-based brick-cache volume rendering.

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
        """Return all GPU resource bindings for the volume block shader.

        Parameters
        ----------
        wobject : Volume
            The volume world object.
        shared : object
            Shared renderer state.
        scene : Scene
            The current scene.

        Returns
        -------
        bindings : dict
            Binding group dictionary.
        """
        geometry = wobject.geometry
        material = wobject.material

        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        cache_view = GfxTextureView(geometry.grid)
        sampler = GfxSampler(material.interpolation, "clamp")
        bindings.append(
            Binding("s_img", "sampler/filtering", sampler, "FRAGMENT")
        )
        bindings.append(
            Binding("t_img", "texture/auto", cache_view, _vertex_and_fragment)
        )

        if material.map is not None:
            bindings.extend(self.define_img_colormap(material.map))

        lut_view = GfxTextureView(material.lut_texture)
        bindings.append(
            Binding("t_lut", "texture/auto", lut_view, "FRAGMENT")
        )

        bindings.append(
            Binding(
                "u_lut_params",
                "buffer/uniform",
                material.lut_params_buffer,
                "FRAGMENT",
                structname="LutParams",
            )
        )

        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)
        return {0: bindings}

    def get_pipeline_info(self, wobject, shared):
        """Return pipeline configuration.

        Parameters
        ----------
        wobject : Volume
            The volume world object.
        shared : object
            Shared renderer state.

        Returns
        -------
        info : dict
            Pipeline info with topology and cull mode.
        """
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.front,
        }

    def get_render_info(self, wobject, shared):
        """Return draw-call parameters.

        Parameters
        ----------
        wobject : Volume
            The volume world object.
        shared : object
            Shared renderer state.

        Returns
        -------
        info : dict
            Indices tuple for the bounding box (36 vertices, 1 instance).
        """
        return {"indices": (36, 1)}

    def get_code(self):
        """Return the WGSL source for this shader.

        Returns
        -------
        code : str
            The volume_block.wgsl source.
        """
        return VOLUME_BLOCK_WGSL
