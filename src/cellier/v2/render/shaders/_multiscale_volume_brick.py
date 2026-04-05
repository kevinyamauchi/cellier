"""Multiscale volume brick shader: material, shader class, and uniform builders.

Kiln-style brick traversal in normalized physical space with nearest-neighbour
sampling, per-brick stochastic jitter, and exact per-axis per-level downscale
factors.

Importing this module registers ``MultiscaleVolumeBrickShader`` as the pygfx
render function for ``(Volume, MultiscaleVolumeBrickMaterial)`` pairs via the
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

_WGSL_PATH = Path(__file__).parent / "wgsl" / "multiscale_volume_brick.wgsl"
_WGSL_SOURCE = _WGSL_PATH.read_text()

_vertex_and_fragment = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT

MAX_LEVELS = 10


# ---------------------------------------------------------------------------
# Uniform dtypes
# ---------------------------------------------------------------------------

VOL_PARAMS_DTYPE = np.dtype(
    [
        ("norm_size_x", "<f4"),
        ("norm_size_y", "<f4"),
        ("norm_size_z", "<f4"),
        ("_pad0", "<f4"),
        ("dataset_size_x", "<f4"),
        ("dataset_size_y", "<f4"),
        ("dataset_size_z", "<f4"),
        ("_pad1", "<f4"),
        ("block_size_x", "<f4"),
        ("block_size_y", "<f4"),
        ("block_size_z", "<f4"),
        ("_pad2", "<f4"),
        ("cache_size_x", "<f4"),
        ("cache_size_y", "<f4"),
        ("cache_size_z", "<f4"),
        ("_pad3", "<f4"),
        ("lut_size_x", "<f4"),
        ("lut_size_y", "<f4"),
        ("lut_size_z", "<f4"),
        ("overlap", "<f4"),
        ("frame_index", "<u4"),
        ("_pad4a", "<u4"),
        ("_pad4b", "<u4"),
        ("_pad4c", "<u4"),
    ]
)

BLOCK_SCALES_DTYPE = np.dtype([(f"scale_{i}", "<f4", (4,)) for i in range(MAX_LEVELS)])


# ---------------------------------------------------------------------------
# Uniform buffer builders
# ---------------------------------------------------------------------------


def compute_normalized_size(
    dataset_size: np.ndarray,
    voxel_spacing: np.ndarray,
) -> np.ndarray:
    """Compute normalized physical extent from dataset size and voxel spacing.

    The longest physical axis is normalized to 1.0.

    Parameters
    ----------
    dataset_size : ndarray, shape (3,)
        Finest-level voxel counts in shader order (x=W, y=H, z=D).
    voxel_spacing : ndarray, shape (3,)
        Physical voxel size in shader order (x=W, y=H, z=D).

    Returns
    -------
    ndarray, shape (3,)
        Normalized physical size in shader order.
    """
    phys_size = dataset_size * voxel_spacing
    return phys_size / phys_size.max()


def build_vol_params_buffer(
    norm_size: np.ndarray,
    dataset_size: np.ndarray,
    base_layout: BlockLayout3D,
    cache_info: BlockCacheParameters3D,
) -> Buffer:
    """Build the VolumeParams uniform buffer.

    Parameters
    ----------
    norm_size : ndarray, shape (3,)
        Normalized physical extent in shader order (x, y, z).
    dataset_size : ndarray, shape (3,)
        Finest-level voxel counts in shader order (x, y, z).
    base_layout : BlockLayout3D
        Layout of the finest level.
    cache_info : BlockCacheParameters3D
        Cache sizing metadata.

    Returns
    -------
    Buffer
        Uniform buffer bound as ``u_vol_params`` in the shader.
    """
    gd, gh, gw = base_layout.grid_dims
    bs = float(base_layout.block_size)
    ov = float(cache_info.overlap)
    cd, ch, cw = cache_info.cache_shape

    data = np.zeros((), dtype=VOL_PARAMS_DTYPE)
    data["norm_size_x"] = float(norm_size[0])
    data["norm_size_y"] = float(norm_size[1])
    data["norm_size_z"] = float(norm_size[2])
    data["dataset_size_x"] = float(dataset_size[0])
    data["dataset_size_y"] = float(dataset_size[1])
    data["dataset_size_z"] = float(dataset_size[2])
    data["block_size_x"] = bs
    data["block_size_y"] = bs
    data["block_size_z"] = bs
    data["cache_size_x"] = float(cw)
    data["cache_size_y"] = float(ch)
    data["cache_size_z"] = float(cd)
    data["lut_size_x"] = float(gw)
    data["lut_size_y"] = float(gh)
    data["lut_size_z"] = float(gd)
    data["overlap"] = ov
    data["frame_index"] = 0

    return Buffer(data, force_contiguous=True)


def build_brick_scales_buffer(
    level_scale_vecs_data: list[np.ndarray],
) -> Buffer:
    """Build the block-scales uniform buffer.

    Unlike the old shader (which stores ``1 / downscale_factor``), this
    stores the **downscale factor** directly.  The shader divides by it.

    Level 0 (1-indexed in the LUT) stores ``[1, 1, 1]``.
    Level k stores the per-axis downscale factor relative to finest.

    Input vectors are in data-axis order ``(sz, sy, sx)``; shader
    fields ``[0], [1], [2]`` are ``(x=W, y=H, z=D)`` so the
    assignment reverses the index.

    Parameters
    ----------
    level_scale_vecs_data : list[np.ndarray]
        Per-level scale vectors in data order, e.g.
        ``[array([1,1,1]), array([1,2,2]), array([1,4,4])]``.

    Returns
    -------
    Buffer
        Uniform buffer bound as ``u_block_scales`` in the shader.
    """
    data = np.zeros((), dtype=BLOCK_SCALES_DTYPE)

    for k in range(1, min(len(level_scale_vecs_data) + 1, MAX_LEVELS)):
        sv = level_scale_vecs_data[k - 1]  # data order: [sz, sy, sx]
        # shader x = W = data axis 2 (sx)
        data[f"scale_{k}"][0] = float(sv[2])
        # shader y = H = data axis 1 (sy)
        data[f"scale_{k}"][1] = float(sv[1])
        # shader z = D = data axis 0 (sz)
        data[f"scale_{k}"][2] = float(sv[0])
        data[f"scale_{k}"][3] = 0.0  # padding

    return Buffer(data, force_contiguous=True)


def compose_world_transform(
    data_to_world: np.ndarray,
    dataset_size: np.ndarray,
    norm_size: np.ndarray,
) -> np.ndarray:
    """Return the 4x4 matrix mapping normalized space -> world space.

    The vertex shader emits proxy-box positions in normalized space.
    ``u_wobject.world_transform`` must go normalized -> world, i.e.::

        world_transform = data_to_world @ norm_to_data

    Parameters
    ----------
    data_to_world : ndarray, shape (4, 4)
        Data-space -> world transform.
    dataset_size : ndarray, shape (3,)
        Finest-level voxel counts (shader order: x, y, z).
    norm_size : ndarray, shape (3,)
        Normalized physical extent (shader order: x, y, z).

    Returns
    -------
    ndarray, shape (4, 4)
        The composed world_transform matrix.
    """
    scale = dataset_size / norm_size
    offset = 0.5 * dataset_size

    norm_to_data = np.eye(4, dtype=np.float64)
    norm_to_data[0, 0] = scale[0]
    norm_to_data[1, 1] = scale[1]
    norm_to_data[2, 2] = scale[2]
    norm_to_data[:3, 3] = offset

    return (data_to_world @ norm_to_data).astype(np.float32)


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------


class MultiscaleVolumeBrickMaterial(gfx.VolumeIsoMaterial):
    """Volume material for the Kiln-style multiscale brick shader.

    Parameters
    ----------
    cache_texture : gfx.Texture
        3D float32 texture — the fixed-size brick cache.
    lut_texture : gfx.Texture
        RGBA8UI 3D texture — the per-brick address lookup table.
    vol_params_buffer : Buffer
        Uniform buffer with normalized size, dataset size, block/cache/LUT
        parameters, and frame index.
    block_scales_buffer : Buffer
        Uniform buffer with per-level downscale factors (10 x vec4).
    clim : tuple[float, float]
        Contrast limits.
    map : gfx.TextureMap, optional
        1D colourmap texture.
    threshold : float
        Isosurface threshold.
    """

    def __init__(
        self,
        cache_texture: gfx.Texture,
        lut_texture: gfx.Texture,
        brick_max_texture: gfx.Texture,
        vol_params_buffer: Buffer,
        block_scales_buffer: Buffer,
        clim: tuple[float, float] = (0.0, 1.0),
        map: gfx.TextureMap | None = None,
        threshold: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(
            clim=clim,
            map=map,
            interpolation="nearest",
            threshold=threshold,
            **kwargs,
        )
        self.cache_texture = cache_texture
        self.lut_texture = lut_texture
        self.brick_max_texture = brick_max_texture
        self.vol_params_buffer = vol_params_buffer
        self.block_scales_buffer = block_scales_buffer
        self.render_mode: str = "iso"
        self._frame_index: int = 0

    def tick(self) -> None:
        """Advance jitter seed. Call once per rendered frame."""
        self._frame_index = (self._frame_index + 1) & 0xFFFF_FFFF
        buf_data = self.vol_params_buffer.data
        buf_data["frame_index"] = self._frame_index
        self.vol_params_buffer.update_full()


# ---------------------------------------------------------------------------
# Shader
# ---------------------------------------------------------------------------


@register_wgpu_render_function(Volume, MultiscaleVolumeBrickMaterial)
class MultiscaleVolumeBrickShader(BaseVolumeShader):
    """Shader for Kiln-style multiscale brick-cache volume rendering."""

    type = "render"

    def get_bindings(self, wobject, shared, scene):
        geometry = wobject.geometry
        material = wobject.material

        # Template variable for debug visualisation modes.
        self["debug_mode"] = getattr(material, "debug_mode", "none")
        # Template variable for render mode (iso / mip).
        self["render_mode"] = getattr(material, "render_mode", "iso")

        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        # Proxy texture — needed for pygfx Volume geometry grid binding.
        proxy_view = GfxTextureView(geometry.grid)
        proxy_sampler = GfxSampler("nearest", "clamp")
        bindings.append(
            Binding("s_img", "sampler/filtering", proxy_sampler, "FRAGMENT")
        )
        bindings.append(
            Binding("t_img", "texture/auto", proxy_view, _vertex_and_fragment)
        )

        # Cache texture + sampler (linear filtering for smooth boundary continuity).
        cache_view = GfxTextureView(material.cache_texture)
        cache_sampler = GfxSampler("linear", "clamp")
        bindings.append(
            Binding("s_cache", "sampler/filtering", cache_sampler, "FRAGMENT")
        )
        bindings.append(Binding("t_cache", "texture/auto", cache_view, "FRAGMENT"))

        # Colourmap — always required for sampled_value_to_color().
        bindings.extend(self.define_img_colormap(material.map))

        # LUT texture.
        lut_view = GfxTextureView(material.lut_texture)
        bindings.append(Binding("t_lut", "texture/auto", lut_view, "FRAGMENT"))

        # Per-brick max intensity texture (R32Float, same grid as LUT).
        brick_max_view = GfxTextureView(material.brick_max_texture)
        bindings.append(
            Binding("t_brick_max", "texture/auto", brick_max_view, "FRAGMENT")
        )

        # Uniform buffers.
        bindings.append(
            Binding(
                "u_vol_params",
                "buffer/uniform",
                material.vol_params_buffer,
                _vertex_and_fragment,
                structname="VolParams",
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
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        return {"indices": (36, 1)}

    def get_code(self):
        return _WGSL_SOURCE
