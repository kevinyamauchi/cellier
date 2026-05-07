"""Label multiscale shaders.

LabelVolumeBrickMaterial + LabelVolumeBrickShader — 3D categorical iso rendering.
LabelBlockMaterial + LabelBlockShader — 2D tile-cache label rendering.

Importing this module registers both shaders with pygfx via
@register_wgpu_render_function.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pygfx as gfx
import wgpu
from pygfx.objects import Image, Volume
from pygfx.renderers.wgpu import (
    Binding,
    GfxSampler,
    GfxTextureView,
    register_wgpu_render_function,
)
from pygfx.renderers.wgpu.shaders.imageshader import ImageShader
from pygfx.renderers.wgpu.shaders.volumeshader import BaseVolumeShader

if TYPE_CHECKING:
    from pygfx.resources import Buffer

_WGSL_DIR = Path(__file__).parent / "wgsl"
_LABEL_VOLUME_BRICK_WGSL = (_WGSL_DIR / "label_volume_brick.wgsl").read_text()
_LABEL_BLOCK_WGSL = (_WGSL_DIR / "label_block.wgsl").read_text()

_vertex_and_fragment = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT


# ---------------------------------------------------------------------------
# 3D material
# ---------------------------------------------------------------------------


class LabelVolumeBrickMaterial(gfx.VolumeBasicMaterial):
    """Volume material for 3D multiscale label rendering (brick cache).

    Parameters
    ----------
    cache_texture : gfx.Texture
        3D int32 texture — the fixed-size brick cache.
    lut_texture : gfx.Texture
        RGBA8UI 3D texture — the per-brick address lookup table.
    brick_max_texture : gfx.Texture
        R32Float 3D texture — per-brick contains_label flag (0.0 / 1.0).
    vol_params_buffer : Buffer
        VolParams uniform buffer.
    block_scales_buffer : Buffer
        BlockScales uniform buffer.
    label_params_buffer : Buffer
        LabelParams uniform buffer (background_label, salt, n_entries).
    label_keys_texture : gfx.Texture | None
        Direct-mode sorted label keys (r32sint).
    label_colors_texture : gfx.Texture | None
        Direct-mode RGBA per entry (rgba32float).
    background_label : int
        Label ID treated as transparent.
    colormap_mode : "random" | "direct"
        Colormap mode (baked into shader; frozen after construction).
    salt : int
        Hash seed for random mode.
    render_mode : "iso_categorical" | "flat_categorical"
        3D render mode.
    """

    def __init__(
        self,
        cache_texture: gfx.Texture,
        lut_texture: gfx.Texture,
        brick_max_texture: gfx.Texture,
        vol_params_buffer: Buffer,
        block_scales_buffer: Buffer,
        label_params_buffer: Buffer,
        label_keys_texture: gfx.Texture | None = None,
        label_colors_texture: gfx.Texture | None = None,
        background_label: int = 0,
        colormap_mode: str = "random",
        salt: int = 0,
        render_mode: str = "iso_categorical",
        n_entries: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cache_texture = cache_texture
        self.lut_texture = lut_texture
        self.brick_max_texture = brick_max_texture
        self.vol_params_buffer = vol_params_buffer
        self.block_scales_buffer = block_scales_buffer
        self.label_params_buffer = label_params_buffer
        self.label_keys_texture = label_keys_texture
        self.label_colors_texture = label_colors_texture
        self.background_label = background_label
        self.colormap_mode = colormap_mode
        self.salt = salt
        self.render_mode = render_mode
        self.n_entries = n_entries


# ---------------------------------------------------------------------------
# 3D shader
# ---------------------------------------------------------------------------


@register_wgpu_render_function(Volume, LabelVolumeBrickMaterial)
class LabelVolumeBrickShader(BaseVolumeShader):
    """Shader for 3D multiscale label brick-cache rendering."""

    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)
        m = wobject.material
        self["colormap_mode"] = m.colormap_mode
        self["render_mode"] = m.render_mode

    def get_bindings(self, wobject, shared, scene):
        geometry = wobject.geometry
        material = wobject.material

        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        # Proxy texture for Volume geometry.
        proxy_view = GfxTextureView(geometry.grid)
        proxy_sampler = GfxSampler("nearest", "clamp")
        bindings.append(
            Binding("s_img", "sampler/filtering", proxy_sampler, "FRAGMENT")
        )
        bindings.append(
            Binding("t_img", "texture/auto", proxy_view, _vertex_and_fragment)
        )

        # Label brick cache (int32, no sampler needed).
        cache_view = GfxTextureView(material.cache_texture)
        bindings.append(Binding("t_cache", "texture/auto", cache_view, "FRAGMENT"))

        # LUT texture (per-brick slot indices).
        lut_view = GfxTextureView(material.lut_texture)
        bindings.append(Binding("t_lut", "texture/auto", lut_view, "FRAGMENT"))

        # Per-brick contains_label flag (R32Float, 0.0 or 1.0).
        brick_max_view = GfxTextureView(material.brick_max_texture)
        bindings.append(
            Binding("t_brick_max", "texture/auto", brick_max_view, "FRAGMENT")
        )

        # Label params uniform.
        bindings.append(
            Binding(
                "u_label_params",
                "buffer/uniform",
                material.label_params_buffer,
                _vertex_and_fragment,
                structname="LabelParams",
            )
        )

        # Vol params uniform.
        bindings.append(
            Binding(
                "u_vol_params",
                "buffer/uniform",
                material.vol_params_buffer,
                _vertex_and_fragment,
                structname="VolParams",
            )
        )

        # Block scales uniform.
        bindings.append(
            Binding(
                "u_block_scales",
                "buffer/uniform",
                material.block_scales_buffer,
                "FRAGMENT",
                structname="BlockScales",
            )
        )

        # Direct-mode LUT textures (only when colormap_mode == "direct").
        if material.label_keys_texture is not None:
            bindings.append(
                Binding(
                    "t_label_keys",
                    "texture/auto",
                    GfxTextureView(material.label_keys_texture),
                    "FRAGMENT",
                )
            )
            bindings.append(
                Binding(
                    "t_label_colors",
                    "texture/auto",
                    GfxTextureView(material.label_colors_texture),
                    "FRAGMENT",
                )
            )

        bindings = dict(enumerate(bindings))
        self.define_bindings(0, bindings)
        return {0: bindings}

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        return {"indices": (36, 1)}

    def get_code(self):
        return _LABEL_VOLUME_BRICK_WGSL


# ---------------------------------------------------------------------------
# 2D material
# ---------------------------------------------------------------------------


class LabelBlockMaterial(gfx.ImageBasicMaterial):
    """Image material for 2D multiscale label tile-cache rendering.

    Parameters
    ----------
    cache_texture : gfx.Texture
        2D int32 texture — the fixed-size tile cache.
    lut_texture : gfx.Texture
        Float32 2D LUT texture (slot indices + level per tile).
    lut_params_buffer : Buffer
        LutParams uniform buffer.
    block_scales_buffer : Buffer
        BlockScales uniform buffer.
    label_params_buffer : Buffer
        LabelParams uniform buffer.
    label_keys_texture : gfx.Texture | None
        Direct-mode sorted label keys.
    label_colors_texture : gfx.Texture | None
        Direct-mode RGBA per entry.
    colormap_mode : "random" | "direct"
        Colormap mode (baked into shader).
    background_label : int
        Label ID treated as transparent.
    """

    def __init__(
        self,
        cache_texture: gfx.Texture,
        lut_texture: gfx.Texture,
        lut_params_buffer: Buffer,
        block_scales_buffer: Buffer,
        label_params_buffer: Buffer,
        label_keys_texture: gfx.Texture | None = None,
        label_colors_texture: gfx.Texture | None = None,
        colormap_mode: str = "random",
        background_label: int = 0,
        n_entries: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(interpolation="nearest", **kwargs)
        self.cache_texture = cache_texture
        self.lut_texture = lut_texture
        self.lut_params_buffer = lut_params_buffer
        self.block_scales_buffer = block_scales_buffer
        self.label_params_buffer = label_params_buffer
        self.label_keys_texture = label_keys_texture
        self.label_colors_texture = label_colors_texture
        self.colormap_mode = colormap_mode
        self.background_label = background_label
        self.n_entries = n_entries


# ---------------------------------------------------------------------------
# 2D shader
# ---------------------------------------------------------------------------


@register_wgpu_render_function(Image, LabelBlockMaterial)
class LabelBlockShader(ImageShader):
    """Shader for 2D multiscale label tile-cache rendering."""

    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)
        m = wobject.material
        self["colormap_mode"] = m.colormap_mode
        self["use_colormap"] = False

    def get_bindings(self, wobject, shared, scene):
        geometry = wobject.geometry
        material = wobject.material

        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        # Proxy texture for Image geometry (drives get_im_geometry quad size).
        proxy_view = GfxTextureView(geometry.grid)
        proxy_sampler = GfxSampler("nearest", "clamp")
        bindings.append(
            Binding("s_img", "sampler/filtering", proxy_sampler, "FRAGMENT")
        )
        bindings.append(
            Binding("t_img", "texture/auto", proxy_view, _vertex_and_fragment)
        )

        # Label tile cache (int32, textureLoad, no sampler).
        cache_view = GfxTextureView(material.cache_texture)
        bindings.append(Binding("t_cache", "texture/auto", cache_view, "FRAGMENT"))

        # LUT texture.
        lut_view = GfxTextureView(material.lut_texture)
        bindings.append(Binding("t_lut", "texture/auto", lut_view, "FRAGMENT"))

        # Label params uniform.
        bindings.append(
            Binding(
                "u_label_params",
                "buffer/uniform",
                material.label_params_buffer,
                "FRAGMENT",
                structname="LabelParams",
            )
        )

        # LUT params uniform.
        bindings.append(
            Binding(
                "u_lut_params",
                "buffer/uniform",
                material.lut_params_buffer,
                "FRAGMENT",
                structname="LutParams",
            )
        )

        # Block scales uniform.
        bindings.append(
            Binding(
                "u_block_scales",
                "buffer/uniform",
                material.block_scales_buffer,
                "FRAGMENT",
                structname="BlockScales",
            )
        )

        # Direct-mode LUT textures.
        if material.label_keys_texture is not None:
            bindings.append(
                Binding(
                    "t_label_keys",
                    "texture/auto",
                    GfxTextureView(material.label_keys_texture),
                    "FRAGMENT",
                )
            )
            bindings.append(
                Binding(
                    "t_label_colors",
                    "texture/auto",
                    GfxTextureView(material.label_colors_texture),
                    "FRAGMENT",
                )
            )

        bindings = dict(enumerate(bindings))
        self.define_bindings(0, bindings)
        return {0: bindings}

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        return {"indices": (6, 1)}

    def get_code(self):
        return _LABEL_BLOCK_WGSL
