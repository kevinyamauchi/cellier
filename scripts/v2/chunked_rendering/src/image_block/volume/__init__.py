"""LUT-based brick-cache volume renderer for pygfx."""

from __future__ import annotations

import time

import numpy as np
import pygfx as gfx
import tensorstore as ts

# Importing shader registers the render function with pygfx.
import image_block.volume.shader as _shader  # noqa: F401
from image_block.core.cache import (
    build_cache_texture,
)
from image_block.core.tile_manager import TileManager
from image_block.volume.cache import compute_cache_info
from image_block.volume.layout import BLOCK_SIZE_DEFAULT, BlockLayout
from image_block.volume.lut import build_lut_texture
from image_block.volume.material import VolumeBlockMaterial
from image_block.volume.state import BlockState3D, BlockVolumeState
from image_block.volume.uniforms import (
    build_block_scales_buffer,
    build_lut_params_buffer,
)

__all__ = [
    "BlockLayout",
    "BlockState3D",
    "BlockVolumeState",  # deprecated alias
    "VolumeBlockMaterial",
    "make_block_volume",
]


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_block_volume(
    ts_stores: list[ts.TensorStore],
    block_size: int = BLOCK_SIZE_DEFAULT,
    gpu_budget_bytes: int = 512 * 1024**2,
    colormap: gfx.TextureMap | None = None,
    clim: tuple[float, float] = (0.0, 1.0),
    threshold: float = 0.5,
) -> tuple[gfx.Volume, BlockState3D]:
    """Create a pygfx Volume using multi-LOAD LUT-based async rendering.

    Parameters
    ----------
    ts_stores : list[ts.TensorStore]
        Pre-opened tensorstore handles ``[finest, ..., coarsest]``.
    block_size : int
        Brick side length in voxels.
    gpu_budget_bytes : int
        Maximum byte budget for the cache texture.
    colormap : gfx.TextureMap, optional
        Colourmap.  Defaults to viridis.
    clim : tuple[float, float]
        Contrast limits.
    threshold : float
        Isosurface threshold.

    Returns
    -------
    vol : gfx.Volume
        A pygfx Volume that can be added to a scene.
    state : BlockState3D
        Mutable state object.
    """
    t0 = time.perf_counter()

    # Derive layouts from each store's shape.
    layouts = [
        BlockLayout.from_volume_shape(
            tuple(int(d) for d in store.domain.shape),
            block_size=block_size,
        )
        for store in ts_stores
    ]
    base_layout = layouts[0]
    n_levels = len(ts_stores)
    print(
        f"  make_block_volume: layouts          {(time.perf_counter()-t0)*1000:.1f} ms"
    )

    t1 = time.perf_counter()
    cache_info = compute_cache_info(block_size, gpu_budget_bytes)
    cache_data, cache_tex = build_cache_texture(cache_info)
    print(
        f"  make_block_volume: cache texture    {(time.perf_counter()-t1)*1000:.1f} ms"
        f"  shape={cache_info.cache_shape}"
    )

    t1 = time.perf_counter()
    lut_data, lut_tex = build_lut_texture(base_layout)
    print(
        f"  make_block_volume: LUT texture      {(time.perf_counter()-t1)*1000:.1f} ms"
        f"  shape={base_layout.grid_dims}"
    )

    tile_manager = TileManager(cache_info)

    lut_params = build_lut_params_buffer(
        base_layout, cache_info, proxy_voxels_per_brick=block_size
    )
    block_scales = build_block_scales_buffer(n_levels)

    if colormap is None:
        colormap = gfx.cm.viridis

    material = VolumeBlockMaterial(
        cache_texture=cache_tex,
        lut_texture=lut_tex,
        lut_params_buffer=lut_params,
        block_scales_buffer=block_scales,
        clim=clim,
        map=colormap,
        threshold=threshold,
    )

    t1 = time.perf_counter()
    gd, gh, gw = base_layout.grid_dims
    proxy_data = np.zeros((gd, gh, gw), dtype=np.float32)
    proxy_tex = gfx.Texture(proxy_data, dim=3)
    print(
        f"  make_block_volume: proxy texture    {(time.perf_counter()-t1)*1000:.1f} ms"
        f"  shape=({gd},{gh},{gw})  [grid dims, not voxel dims]"
    )

    t1 = time.perf_counter()
    geometry = gfx.Geometry(grid=proxy_tex)
    vol = gfx.Volume(geometry, material)

    vol.local.scale = (float(block_size), float(block_size), float(block_size))
    _off = 0.5 * (float(block_size) - 1.0)
    vol.local.position = (_off, _off, _off)
    print(
        f"  make_block_volume: volume object    {(time.perf_counter()-t1)*1000:.1f} ms"
    )
    print(
        f"  make_block_volume: TOTAL            {(time.perf_counter()-t0)*1000:.1f} ms"
    )

    state = BlockState3D(
        ts_stores=ts_stores,
        layouts=layouts,
        cache_info=cache_info,
        cache_data=cache_data,
        cache_tex=cache_tex,
        lut_data=lut_data,
        lut_tex=lut_tex,
        tile_manager=tile_manager,
        block_size=block_size,
        overlap=cache_info.overlap,
    )

    return vol, state
