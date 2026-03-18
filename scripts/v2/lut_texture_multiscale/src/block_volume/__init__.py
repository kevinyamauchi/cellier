"""LUT-based brick-cache volume renderer for pygfx.

Phase 1: multi-LOD renderer with a fixed-size GPU cache, LRU eviction,
and per-frame LUT rebuild.  Bricks from any LOD level can occupy any
cache slot.

The geometry uses a *proxy* texture whose dimensions match the
volume's padded shape so that ``get_vol_geometry()`` produces the
correct bounding box.  The actual brick data lives in a separate
*cache* texture bound as ``t_cache``.
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx

from block_volume.cache import (
    CacheInfo,
    build_cache_texture,
    commit_brick,
    compute_cache_info,
)
from block_volume.layout import BLOCK_SIZE_DEFAULT, BlockLayout
from block_volume.lod import select_levels
from block_volume.lut import build_lut_texture, rebuild_lut
from block_volume.material import VolumeBlockMaterial
from block_volume.tile_manager import BrickKey, TileManager
from block_volume.uniforms import (
    build_block_scales_buffer,
    build_lut_params_buffer,
)

# Importing shader registers the render function with pygfx.
import block_volume.shader as _shader  # noqa: F401

__all__ = [
    "BlockLayout",
    "BlockVolumeState",
    "VolumeBlockMaterial",
    "make_block_volume",
]


class BlockVolumeState:
    """Holds the mutable state for a multi-LOD block volume.

    Created by :func:`make_block_volume` and exposed so that the
    example can trigger per-frame LOD + cache updates.

    Attributes
    ----------
    levels : list[np.ndarray]
        CPU-side multiscale arrays ``[finest, ..., coarsest]``.
    layouts : list[BlockLayout]
        One ``BlockLayout`` per level.
    base_layout : BlockLayout
        Layout of the finest level (``layouts[0]``).
    cache_info : CacheInfo
        Cache sizing metadata.
    cache_data : np.ndarray
        CPU-side backing array for the cache texture.
    cache_tex : gfx.Texture
        GPU cache texture.
    lut_data : np.ndarray
        CPU-side backing array for the LUT texture.
    lut_tex : gfx.Texture
        GPU LUT texture.
    tile_manager : TileManager
        Brick-to-slot mapping with LRU eviction.
    block_size : int
        Brick side length in voxels.
    n_levels : int
        Number of LOD levels.
    frame_number : int
        Monotonically increasing frame counter.
    """

    def __init__(
        self,
        levels: list[np.ndarray],
        layouts: list[BlockLayout],
        cache_info: CacheInfo,
        cache_data: np.ndarray,
        cache_tex: gfx.Texture,
        lut_data: np.ndarray,
        lut_tex: gfx.Texture,
        tile_manager: TileManager,
        block_size: int,
        overlap: int = 1,
    ) -> None:
        self.levels = levels
        self.layouts = layouts
        self.base_layout = layouts[0]
        self.cache_info = cache_info
        self.cache_data = cache_data
        self.cache_tex = cache_tex
        self.lut_data = lut_data
        self.lut_tex = lut_tex
        self.tile_manager = tile_manager
        self.block_size = block_size
        self.overlap = overlap
        self.n_levels = len(levels)
        self.frame_number = 0

    def update(
        self,
        camera_pos: np.ndarray,
        thresholds: list[float] | None = None,
        force_level: int | None = None,
    ) -> dict:
        """Run one update cycle: LOD select → stage → commit → LUT rebuild.

        Parameters
        ----------
        camera_pos : np.ndarray
            Camera world-space position ``(x, y, z)``.
        thresholds : list[float] or None
            Distance cutoffs for LOD transitions.
        force_level : int or None
            If set, bypass LOD selection and assign every base-grid
            brick to this level (1 = finest).  Useful for debugging.

        Returns
        -------
        stats : dict
            ``{"hits": int, "misses": int, "fills": int}``.
        """
        self.frame_number += 1

        # 1. LOD selection (or forced level).
        if force_level is not None:
            gd, gh, gw = self.base_layout.grid_dims
            required: dict[BrickKey, int] = {}
            for gz in range(gd):
                for gy in range(gh):
                    for gx in range(gw):
                        scale = 2 ** (force_level - 1)
                        key = BrickKey(
                            level=force_level,
                            gz=gz // scale,
                            gy=gy // scale,
                            gx=gx // scale,
                        )
                        required[key] = force_level
        else:
            required = select_levels(
                self.base_layout,
                self.n_levels,
                camera_pos,
                thresholds=thresholds,
            )

        # 2. Sort bricks nearest-to-camera first so that close bricks
        #    claim cache slots before far ones.  If the cache is full,
        #    far bricks will be the ones that fail to load or evict
        #    each other, while near bricks stay resident.
        bs = self.block_size

        def _brick_distance(item: tuple) -> float:
            key = item[0]
            scale = 2 ** (key.level - 1)
            cx = (key.gx + 0.5) * bs * scale
            cy = (key.gy + 0.5) * bs * scale
            cz = (key.gz + 0.5) * bs * scale
            centre = np.array([cx, cy, cz])
            return float(np.linalg.norm(centre - camera_pos))

        sorted_required = dict(sorted(required.items(), key=_brick_distance))

        # 3. Stage (find hits and misses, plan fills).
        fill_plan = self.tile_manager.stage(sorted_required, self.frame_number)

        # 4. Commit brick data for each fill.
        for brick_key, slot in fill_plan:
            data = self._read_brick(brick_key)
            commit_brick(
                self.cache_data,
                self.cache_tex,
                slot.grid_pos,
                self.cache_info.padded_block_size,
                data,
            )

        # 5. Rebuild LUT.
        rebuild_lut(
            self.base_layout,
            self.tile_manager,
            self.n_levels,
            self.lut_data,
            self.lut_tex,
        )

        # 6. Diagnostics.
        n_required = len(required)
        n_fills = len(fill_plan)

        # LUT level breakdown.
        level_counts: dict[int, int] = {}
        gd, gh, gw = self.base_layout.grid_dims
        for gz in range(gd):
            for gy in range(gh):
                for gx in range(gw):
                    lv = int(self.lut_data[gz, gy, gx, 3])
                    level_counts[lv] = level_counts.get(lv, 0) + 1

        print(f"  camera world pos: ({camera_pos[0]:.1f}, {camera_pos[1]:.1f}, {camera_pos[2]:.1f})")
        if force_level is not None:
            print(f"  force_level={force_level}")
        print(f"  required={n_required}  hits={n_required - n_fills}  fills={n_fills}")
        print(f"  LUT level breakdown: {dict(sorted(level_counts.items()))}")
        print(f"  tilemap size: {len(self.tile_manager.tilemap)}  "
              f"free_slots: {len(self.tile_manager.free_slots)}")

        # Sample a few LUT entries for sanity.
        for gz, gy, gx in [(0, 0, 0), (0, 0, gw - 1), (gd - 1, gh - 1, gw - 1)]:
            entry = self.lut_data[gz, gy, gx]
            print(f"  LUT[{gz},{gy},{gx}] = (sx={entry[0]}, sy={entry[1]}, "
                  f"sz={entry[2]}, level={entry[3]})")

        return {
            "hits": n_required - n_fills,
            "misses": n_fills,
            "fills": n_fills,
            "total_required": n_required,
        }

    def _read_brick(self, key: BrickKey) -> np.ndarray:
        """Extract a brick with overlap border from the CPU-side level arrays.

        The returned array is ``(padded, padded, padded)`` where
        ``padded = block_size + 2 * overlap``.  The border voxels are
        copied from spatial neighbours so the GPU's linear interpolation
        at brick edges reads correct data instead of bleeding across
        unrelated cache slots.

        Parameters
        ----------
        key : BrickKey
            Brick identifier (level and grid position).

        Returns
        -------
        data : np.ndarray
            Float32 array of shape ``(padded, padded, padded)``.
        """
        level_idx = key.level - 1  # 0-indexed into self.levels
        vol = self.levels[level_idx]
        bs = self.block_size
        ov = self.overlap
        padded = bs + 2 * ov

        # Origin of the padded region (can be negative).
        z0 = key.gz * bs - ov
        y0 = key.gy * bs - ov
        x0 = key.gx * bs - ov

        brick = np.zeros((padded, padded, padded), dtype=np.float32)
        d, h, w = vol.shape

        # Source region clamped to valid bounds.
        sz0, sy0, sx0 = max(z0, 0), max(y0, 0), max(x0, 0)
        sz1 = min(z0 + padded, d)
        sy1 = min(y0 + padded, h)
        sx1 = min(x0 + padded, w)

        if sz1 > sz0 and sy1 > sy0 and sx1 > sx0:
            # Destination offsets into the brick array.
            dz0 = sz0 - z0
            dy0 = sy0 - y0
            dx0 = sx0 - x0
            dz1 = dz0 + (sz1 - sz0)
            dy1 = dy0 + (sy1 - sy0)
            dx1 = dx0 + (sx1 - sx0)
            brick[dz0:dz1, dy0:dy1, dx0:dx1] = vol[sz0:sz1, sy0:sy1, sx0:sx1]

        return brick


def make_block_volume(
    levels: list[np.ndarray],
    block_size: int = BLOCK_SIZE_DEFAULT,
    gpu_budget_bytes: int = 512 * 1024**2,
    colormap: gfx.TextureMap | None = None,
    clim: tuple[float, float] = (0.0, 1.0),
    threshold: float = 0.5,
) -> tuple[gfx.Volume, BlockVolumeState]:
    """Create a pygfx Volume using multi-LOD LUT-based rendering.

    Parameters
    ----------
    levels : list[np.ndarray]
        Multiscale pyramid ``[finest, ..., coarsest]``.
        Each array is float32 ``(D, H, W)``.
    block_size : int
        Brick side length in voxels.
    gpu_budget_bytes : int
        Maximum byte budget for the cache texture.
    colormap : gfx.TextureMap, optional
        Colourmap. Defaults to viridis.
    clim : tuple[float, float]
        Contrast limits.
    threshold : float
        Isosurface threshold.

    Returns
    -------
    vol : gfx.Volume
        A pygfx Volume that can be added to a scene.
    state : BlockVolumeState
        Mutable state object — call ``state.update(camera_pos)`` each
        frame to drive LOD selection, cache management, and LUT rebuild.
    """
    layouts = [
        BlockLayout.from_volume_shape(lv.shape, block_size=block_size)
        for lv in levels
    ]
    base_layout = layouts[0]
    n_levels = len(levels)

    cache_info = compute_cache_info(block_size, gpu_budget_bytes)
    cache_data, cache_tex = build_cache_texture(cache_info)
    lut_data, lut_tex = build_lut_texture(base_layout)

    tile_manager = TileManager(cache_info)

    lut_params = build_lut_params_buffer(base_layout, cache_info)
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

    # Proxy texture: its only role is to give get_vol_geometry() the
    # correct textureDimensions for the volume bounding box.
    # Shape = base level's padded dimensions in numpy (D, H, W) order.
    proxy_data = np.zeros(base_layout.padded_shape, dtype=np.float32)
    proxy_tex = gfx.Texture(proxy_data, dim=3)

    geometry = gfx.Geometry(grid=proxy_tex)
    vol = gfx.Volume(geometry, material)

    state = BlockVolumeState(
        levels=levels,
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
