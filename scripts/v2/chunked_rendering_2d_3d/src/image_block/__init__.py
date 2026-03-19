"""LUT-based tile-cache 2D image renderer for pygfx.

Async multiscale 2D image rendering with LUT indirection and a
fixed-size GPU tile cache, backed by tensorstore zarr I/O and driven
by PySide6.QtAsyncio.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
import pygfx as gfx
import tensorstore as ts

from image_block.material import ImageBlockMaterial
from image_block.shared.cache import (
    CacheInfo,
    build_cache_texture,
    compute_cache_info,
)
from image_block.shared.layout import BLOCK_SIZE_DEFAULT, BlockLayout2D
from image_block.shared.lut import build_lut_texture
from image_block.shared.state import BlockState2D
from image_block.shared.tile_manager import TileManager
from image_block.shared.uniforms import (
    build_block_scales_buffer,
    build_lut_params_buffer,
)

# Importing shader registers the render function with pygfx.
import image_block.shader as _shader  # noqa: F401

__all__ = [
    "BlockLayout2D",
    "BlockState2D",
    "ImageBlockMaterial",
    "make_block_image",
]


# ---------------------------------------------------------------------------
# Store initialisation
# ---------------------------------------------------------------------------


def _detect_zarr_driver(level_path: pathlib.Path) -> str:
    """Return the correct tensorstore driver for a zarr level directory.

    Detects format by the metadata sentinel file:
    - ``.zarray``   -> zarr v2  -> driver ``"zarr"``
    - ``zarr.json`` -> zarr v3  -> driver ``"zarr3"``
    """
    p = pathlib.Path(level_path)
    if (p / ".zarray").exists():
        return "zarr"
    if (p / "zarr.json").exists():
        return "zarr3"
    raise FileNotFoundError(
        f"Cannot determine zarr format for '{p}': "
        f"neither '.zarray' (zarr v2) nor 'zarr.json' (zarr v3) found.\n"
        f"Re-run with --make-files to regenerate the store."
    )


def open_ts_stores(
    zarr_path: pathlib.Path,
    scale_names: list[str],
) -> list[ts.TensorStore]:
    """Open one tensorstore per scale level (read-only, synchronous).

    Auto-detects zarr v2 vs v3 format.
    Must be called **before** ``QtAsyncio.run()`` starts the event loop.

    Parameters
    ----------
    zarr_path : pathlib.Path
        Root directory of the multiscale zarr store.
    scale_names : list[str]
        Subdirectory names in order finest -> coarsest.

    Returns
    -------
    stores : list[ts.TensorStore]
        One open store per scale level.
    """
    stores = []
    for name in scale_names:
        level_path = pathlib.Path(zarr_path) / name
        driver = _detect_zarr_driver(level_path)
        spec = {
            "driver": driver,
            "kvstore": {
                "driver": "file",
                "path": str(level_path),
            },
        }
        store = ts.open(spec).result()
        shape_str = tuple(int(d) for d in store.domain.shape)
        fmt = "v2" if driver == "zarr" else "v3"
        print(f"  {name}: opened as zarr {fmt}  shape={shape_str}")
        stores.append(store)
    return stores


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_block_image(
    ts_stores: list[ts.TensorStore],
    block_size: int = BLOCK_SIZE_DEFAULT,
    gpu_budget_bytes: int = 512 * 1024**2,
    colormap: gfx.TextureMap | None = None,
    clim: tuple[float, float] = (0.0, 1.0),
    overlap: int = 1,
    z_slice: int | None = None,
) -> tuple[gfx.Image, BlockState2D]:
    """Create a pygfx Image using multi-LOD LUT-based async 2D rendering.

    Parameters
    ----------
    ts_stores : list[ts.TensorStore]
        Pre-opened tensorstore handles ``[finest, ..., coarsest]``.
    block_size : int
        Tile side length in pixels.
    gpu_budget_bytes : int
        Maximum byte budget for the cache texture.
    colormap : gfx.TextureMap, optional
        Colourmap.  Defaults to viridis.
    clim : tuple[float, float]
        Contrast limits.
    overlap : int
        Border pixels per tile side.
    z_slice : int or None
        Z-slice index for 3D stores.  None = mid-slice.

    Returns
    -------
    image : gfx.Image
        A pygfx Image that can be added to a scene.
    state : BlockState2D
        Mutable state object.
    """
    t0 = time.perf_counter()

    # 1. Derive layouts from each store's shape.
    layouts = []
    for store in ts_stores:
        shape = tuple(int(d) for d in store.domain.shape)
        if len(shape) == 3:
            # 3D store: use (Y, X) = shape[1:]
            image_shape = (shape[1], shape[2])
        elif len(shape) == 2:
            image_shape = shape
        else:
            raise ValueError(f"Unexpected store shape: {shape}")
        layouts.append(
            BlockLayout2D.from_shape(image_shape, block_size=block_size, overlap=overlap)
        )

    base_layout = layouts[0]
    n_levels = len(ts_stores)
    gh, gw = base_layout.grid_dims

    print(f"  base layout: {base_layout.volume_shape} -> "
          f"grid {gh}x{gw} = {base_layout.n_tiles} tiles")

    # 2. Compute cache info.
    cache_info = compute_cache_info(gpu_budget_bytes, block_size, overlap)
    cache_data, cache_tex = build_cache_texture(cache_info)
    pbs = cache_info.padded_block_size
    gs = cache_info.grid_side
    cache_pixels = gs * pbs
    cache_mb = cache_pixels * cache_pixels * 4 / (1024**2)
    print(f"  cache: {gs}x{gs} slots = {cache_info.n_slots} slots, "
          f"texture {cache_pixels}x{cache_pixels} ({cache_mb:.1f} MB)")

    # 3. Build LUT texture.
    lut_data, lut_tex = build_lut_texture(base_layout.grid_dims)

    # 4. Build uniform buffers.
    lut_params = build_lut_params_buffer(base_layout, cache_info)
    block_scales = build_block_scales_buffer(n_levels)

    # 5. Tile manager.
    tile_manager = TileManager(cache_info)

    # 6. Build proxy texture: (gH, gW), one texel per tile at finest grid.
    proxy_data = np.zeros((gh, gw), dtype=np.float32)
    proxy_tex = gfx.Texture(proxy_data, dim=2)

    # 7. Build material.
    if colormap is None:
        colormap = gfx.cm.viridis

    material = ImageBlockMaterial(
        cache_texture=cache_tex,
        lut_texture=lut_tex,
        lut_params_buffer=lut_params,
        block_scales_buffer=block_scales,
        clim=clim,
        map=colormap,
    )

    # 8. Build Image.
    geometry = gfx.Geometry(grid=proxy_tex)
    image = gfx.Image(geometry, material)

    # 9. Scale: proxy has 1 texel per tile, so scale by block_size
    #    to make the quad cover the correct world-space extent.
    image.local.scale = (block_size, block_size, 1.0)

    # get_im_geometry() positions the quad from (-0.5, -0.5) to
    # (gW-0.5, gH-0.5) in proxy texels.  After scaling by block_size
    # the quad spans (-bs/2, -bs/2) to (gW*bs - bs/2, gH*bs - bs/2).
    # Shift by +bs/2 so it sits at (0, 0) to (gW*bs, gH*bs).
    image.local.position = (block_size * 0.5, block_size * 0.5, 0)

    # 10. Construct state.
    state = BlockState2D(
        ts_stores=ts_stores,
        layouts=layouts,
        cache_info=cache_info,
        cache_data=cache_data,
        cache_tex=cache_tex,
        lut_data=lut_data,
        lut_tex=lut_tex,
        tile_manager=tile_manager,
        block_size=block_size,
        overlap=overlap,
        z_slice=z_slice,
    )

    t_total = (time.perf_counter() - t0) * 1000
    print(f"  make_block_image: {t_total:.1f} ms")

    return image, state
