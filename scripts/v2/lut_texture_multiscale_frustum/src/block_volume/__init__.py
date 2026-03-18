"""LUT-based brick-cache volume renderer for pygfx.

Phase 2: adds optional frustum culling to ``BlockVolumeState.update()``
via the new ``frustum_planes`` parameter, plus detailed timing
instrumentation in the returned stats dict.

All rendering internals (cache, LUT, shader, WGSL) are unchanged from
Phase 1.
"""

from __future__ import annotations

import time

import numpy as np
import pygfx as gfx

from block_volume.cache import (
    CacheInfo,
    build_cache_texture,
    commit_brick,
    compute_cache_info,
)
from block_volume.frustum import bricks_in_frustum
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
    caller can trigger explicit LOD + cache updates via ``update()``.

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
        Monotonically increasing update counter.
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
        frustum_planes: np.ndarray | None = None,
    ) -> dict:
        """Run one update cycle: LOD select → [frustum cull] → stage → commit → LUT rebuild.

        Parameters
        ----------
        camera_pos : np.ndarray
            Camera world-space position ``(x, y, z)``.
        thresholds : list[float] or None
            Distance cutoffs for LOD transitions.
        force_level : int or None
            If set, bypass LOD selection and assign every base-grid
            brick to this level (1 = finest).  Useful for debugging.
        frustum_planes : ndarray, shape (6, 4) or None
            Inward-pointing half-space planes from
            ``frustum.frustum_planes_from_corners``.  When provided,
            bricks whose AABB is entirely outside the frustum are
            excluded from the required set before staging.

        Returns
        -------
        stats : dict
            Timing and diagnostic information::

                hits            int   cache hits
                misses          int   cache misses
                fills           int   bricks uploaded this cycle
                total_required  int   bricks before culling
                n_culled        int   bricks removed by frustum cull (0 if disabled)
                level_counts    dict  {level: count} from the LUT
                lod_select_ms   float
                distance_sort_ms float
                frustum_cull_ms float  (0.0 if frustum_planes is None)
                stage_ms        float
                commit_ms       float
                lut_rebuild_ms  float
                update_total_ms float
        """
        t_update_start = time.perf_counter()
        self.frame_number += 1

        # ── 1. LOD selection (or forced level) ────────────────────────
        t0 = time.perf_counter()
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
        lod_select_ms = (time.perf_counter() - t0) * 1000

        # ── 2. Sort nearest-to-camera first ───────────────────────────
        t0 = time.perf_counter()
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
        distance_sort_ms = (time.perf_counter() - t0) * 1000

        n_total = len(sorted_required)

        # ── 3. Frustum cull (optional) ────────────────────────────────
        cull_timings: dict = {}
        n_culled = 0
        frustum_cull_ms = 0.0

        if frustum_planes is not None:
            t0 = time.perf_counter()
            sorted_required, cull_timings = bricks_in_frustum(
                sorted_required, self.base_layout, self.block_size, frustum_planes
            )
            frustum_cull_ms = (time.perf_counter() - t0) * 1000
            n_culled = n_total - len(sorted_required)

        # ── 4. Truncate to cache budget (nearest-first) ───────────────
        # Bricks beyond the slot count are dropped before staging.
        # The LUT rebuild's existing fallback chain automatically displays
        # their coarsest cached ancestor — no special handling needed.
        n_needed = len(sorted_required)
        n_budget = self.cache_info.n_slots
        n_dropped = max(0, n_needed - n_budget)
        if n_dropped:
            sorted_required = dict(list(sorted_required.items())[:n_budget])
        print(f"chunks: {n_needed} needed / {n_budget} cache slots  ({n_dropped} dropped)")

        # ── 4. Stage (find hits and misses, plan fills) ───────────────
        t0 = time.perf_counter()
        fill_plan = self.tile_manager.stage(sorted_required, self.frame_number)
        stage_ms = (time.perf_counter() - t0) * 1000

        # ── 5. Commit brick data for each fill ────────────────────────
        t0 = time.perf_counter()
        for brick_key, slot in fill_plan:
            data = self._read_brick(brick_key)
            commit_brick(
                self.cache_data,
                self.cache_tex,
                slot.grid_pos,
                self.cache_info.padded_block_size,
                data,
            )
        commit_ms = (time.perf_counter() - t0) * 1000

        # ── 6. Rebuild LUT ────────────────────────────────────────────
        t0 = time.perf_counter()
        rebuild_lut(
            self.base_layout,
            self.tile_manager,
            self.n_levels,
            self.lut_data,
            self.lut_tex,
        )
        lut_rebuild_ms = (time.perf_counter() - t0) * 1000

        # ── 7. Diagnostics ────────────────────────────────────────────
        n_required = len(sorted_required)
        n_fills = len(fill_plan)

        # LUT level breakdown.
        level_counts: dict[int, int] = {}
        gd, gh, gw = self.base_layout.grid_dims
        for gz in range(gd):
            for gy in range(gh):
                for gx in range(gw):
                    lv = int(self.lut_data[gz, gy, gx, 3])
                    level_counts[lv] = level_counts.get(lv, 0) + 1

        update_total_ms = (time.perf_counter() - t_update_start) * 1000

        return {
            "hits": n_required - n_fills,
            "misses": n_fills,
            "fills": n_fills,
            "total_required": n_total,
            "n_culled": n_culled,
            "n_needed": n_needed,
            "n_budget": n_budget,
            "n_dropped": n_dropped,
            "level_counts": level_counts,
            "cull_timings": cull_timings,
            "lod_select_ms": lod_select_ms,
            "distance_sort_ms": distance_sort_ms,
            "frustum_cull_ms": frustum_cull_ms,
            "stage_ms": stage_ms,
            "commit_ms": commit_ms,
            "lut_rebuild_ms": lut_rebuild_ms,
            "update_total_ms": update_total_ms,
        }

    def _read_brick(self, key: BrickKey) -> np.ndarray:
        """Extract a brick with overlap border from the CPU-side level arrays.

        The returned array is ``(padded, padded, padded)`` where
        ``padded = block_size + 2 * overlap``.

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
        Mutable state object — call ``state.update(camera_pos)`` to
        drive LOD selection, cache management, and LUT rebuild.
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
