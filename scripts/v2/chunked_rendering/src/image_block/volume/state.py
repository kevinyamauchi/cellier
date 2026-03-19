"""BlockState3D: mutable state for multi-LOAD 3D brick-cache volume rendering."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pygfx as gfx
import tensorstore as ts

from image_block.core.block_key import BlockKey
from image_block.core.cache import CacheInfo
from image_block.core.state_base import BlockStateBase
from image_block.core.tile_manager import TileManager, TileSlot
from image_block.volume.frustum import bricks_in_frustum_arr
from image_block.volume.layout import BlockLayout
from image_block.volume.load import (
    arr_to_brick_keys,
    build_level_grids,
    select_levels_arr_forced,
    select_levels_from_cache,
    sort_arr_by_distance,
)
from image_block.volume.lut import rebuild_lut


class BlockState3D(BlockStateBase):
    """Mutable state for a multi-LOAD 3D brick-cache volume.

    Renamed from BlockVolumeState.  Importable as::

        from image_block import BlockState3D

    Attributes
    ----------
    ts_stores : list[ts.TensorStore]
        Open tensorstore handles ``[finest, ..., coarsest]``.
        Chunk data is read lazily, one brick at a time.
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
        Number of LOAD levels.
    frame_number : int
        Monotonically increasing update counter.
    _level_grids : list[dict]
        Precomputed per-level coarse grid arrays (static after init).
        See ``load.build_level_grids`` for structure.
    """

    def __init__(
        self,
        ts_stores: list[ts.TensorStore],
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
        self.ts_stores = ts_stores
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
        self.n_levels = len(ts_stores)
        self.frame_number = 0
        # Precompute static coarse grid arrays for each LOAD level.
        self._level_grids = build_level_grids(layouts[0], self.n_levels)
        total_bricks = sum(len(g["arr"]) for g in self._level_grids)
        print(
            f"  LOAD grid cache: {self.n_levels} levels, "
            f"{total_bricks} total coarse bricks cached"
        )

    # ------------------------------------------------------------------
    # Synchronous planning phase
    # ------------------------------------------------------------------

    def plan_update(
        self,
        camera_pos: np.ndarray,
        thresholds: list[float] | None = None,
        force_level: int | None = None,
        frustum_planes: np.ndarray | None = None,
    ) -> tuple[list[tuple[BlockKey, TileSlot]], dict]:
        """Run the synchronous planning phase.

        Executes: vectorised LOAD select → vectorised distance sort →
        frustum cull → budget truncation → stage().

        Parameters
        ----------
        camera_pos : np.ndarray
            Camera world-space position ``(x, y, z)``.
        thresholds : list[float] or None
            Screen-space LOAD distance cutoffs.
        force_level : int or None
            Override all bricks to this LOAD level.
        frustum_planes : ndarray, shape (6, 4) or None
            Inward-pointing frustum half-space planes.

        Returns
        -------
        fill_plan : list[tuple[BlockKey, TileSlot]]
            Ordered nearest-first.  Empty if all bricks are cache hits.
        stats : dict
            Timing and diagnostic information.
        """
        t_plan_start = time.perf_counter()
        self.frame_number += 1

        # ── 1. LOAD selection ──────────────────────────────────────────
        t0 = time.perf_counter()
        if force_level is not None:
            brick_arr = select_levels_arr_forced(
                self.base_layout, force_level, self._level_grids
            )
        else:
            brick_arr = select_levels_from_cache(
                self._level_grids,
                self.n_levels,
                camera_pos,
                thresholds=thresholds,
                base_layout=self.base_layout,
            )
        load_select_ms = (time.perf_counter() - t0) * 1000

        # ── 2. Distance sort ──────────────────────────────────────────
        t0 = time.perf_counter()
        brick_arr = sort_arr_by_distance(
            brick_arr, camera_pos, self.block_size, self._level_grids
        )
        distance_sort_ms = (time.perf_counter() - t0) * 1000

        n_total = len(brick_arr)

        # ── 3. Frustum cull (optional) ────────────────────────────────
        cull_timings: dict = {}
        n_culled = 0
        frustum_cull_ms = 0.0

        if frustum_planes is not None:
            t0 = time.perf_counter()
            brick_arr, cull_timings = bricks_in_frustum_arr(
                brick_arr, self.block_size, frustum_planes
            )
            frustum_cull_ms = (time.perf_counter() - t0) * 1000
            n_culled = n_total - len(brick_arr)

        # ── 4. Truncate to cache budget ───────────────────────────────
        n_needed = len(brick_arr)
        n_budget = self.cache_info.n_slots
        n_dropped = max(0, n_needed - n_budget)
        if n_dropped:
            brick_arr = brick_arr[:n_budget]
        print(
            f"chunks: {n_needed} needed / {n_budget} cache slots  ({n_dropped} dropped)"
        )

        # ── 5. Convert to BlockKey dict ───────────────────────────────
        t0 = time.perf_counter()
        sorted_required = arr_to_brick_keys(brick_arr)

        # ── 6. Stage (find hits and misses, plan fills) ───────────────
        fill_plan = self.tile_manager.stage(sorted_required, self.frame_number)
        stage_ms = (time.perf_counter() - t0) * 1000

        plan_total_ms = (time.perf_counter() - t_plan_start) * 1000

        # ── Fill-plan order debug (first 10 misses) ───────────────────
        if fill_plan:
            cam = np.asarray(camera_pos, dtype=np.float64)
            bs = self.block_size
            print("  fill_plan order (first 10 misses, nearest-first expected):")
            for i, (bk, _slot) in enumerate(fill_plan[:10]):
                scale = 2 ** (bk.level - 1)
                cx = (bk.gx + 0.5) * bs * scale
                cy = (bk.gy + 0.5) * bs * scale
                cz = (bk.gz + 0.5) * bs * scale
                dist = float(np.linalg.norm(np.array([cx, cy, cz]) - cam))
                print(
                    f"    [{i:3d}] L{bk.level} ({bk.gz:3d},{bk.gy:3d},{bk.gx:3d})"
                    f"  dist={dist:8.1f}"
                )

        # LUT level breakdown
        level_counts: dict[int, int] = {}
        gd, gh, gw = self.base_layout.grid_dims
        for gz in range(gd):
            for gy in range(gh):
                for gx in range(gw):
                    lv = int(self.lut_data[gz, gy, gx, 3])
                    level_counts[lv] = level_counts.get(lv, 0) + 1

        stats = {
            "hits": len(sorted_required) - len(fill_plan),
            "misses": len(fill_plan),
            "fills": len(fill_plan),
            "total_required": n_total,
            "n_culled": n_culled,
            "n_needed": n_needed,
            "n_budget": n_budget,
            "n_dropped": n_dropped,
            "level_counts": level_counts,
            "cull_timings": cull_timings,
            "load_select_ms": load_select_ms,
            "distance_sort_ms": distance_sort_ms,
            "frustum_cull_ms": frustum_cull_ms,
            "stage_ms": stage_ms,
            "plan_total_ms": plan_total_ms,
        }
        return fill_plan, stats

    # ------------------------------------------------------------------
    # Abstract method implementations (BlockStateBase)
    # ------------------------------------------------------------------

    async def _read_block_async(self, key: BlockKey) -> np.ndarray:
        """Read one padded brick from the tensorstore zarr store.

        Parameters
        ----------
        key : BlockKey
            Identifies the brick by level and grid position.

        Returns
        -------
        brick : np.ndarray
            Float32 array of shape ``(padded_bs, padded_bs, padded_bs)``.
        """
        store = self.ts_stores[key.level - 1]
        bs = self.block_size
        ov = self.overlap
        padded = bs + 2 * ov

        # True (possibly negative) padded-region origin.
        z0 = key.gz * bs - ov
        y0 = key.gy * bs - ov
        x0 = key.gx * bs - ov

        # Output brick (zeroed — boundary regions stay zero).
        brick = np.zeros((padded, padded, padded), dtype=np.float32)

        # Source region clamped to valid store bounds.
        sd, sh, sw = (int(d) for d in store.domain.shape)
        sz0 = max(z0, 0)
        sz1 = min(z0 + padded, sd)
        sy0 = max(y0, 0)
        sy1 = min(y0 + padded, sh)
        sx0 = max(x0, 0)
        sx1 = min(x0 + padded, sw)

        if sz1 > sz0 and sy1 > sy0 and sx1 > sx0:
            # Destination offsets into the brick array.
            dz0 = sz0 - z0
            dz1 = dz0 + (sz1 - sz0)
            dy0 = sy0 - y0
            dy1 = dy0 + (sy1 - sy0)
            dx0 = sx0 - x0
            dx1 = dx0 + (sx1 - sx0)

            region = np.asarray(
                await store[sz0:sz1, sy0:sy1, sx0:sx1].read(),
                dtype=np.float32,
            )
            brick[dz0:dz1, dy0:dy1, dx0:dx1] = region

        return brick

    def _rebuild_lut(self) -> None:
        """Rewrite lut_data from tile_manager.tilemap and upload to GPU."""
        rebuild_lut(
            self.base_layout,
            self.tile_manager,
            self.n_levels,
            self.lut_data,
            self.lut_tex,
        )

    # ------------------------------------------------------------------
    # Public async commit method
    # ------------------------------------------------------------------

    async def commit_bricks_async(
        self,
        fill_plan: list[tuple[BlockKey, TileSlot]],
        status_callback: Callable[[str], None] | None = None,
        batch_size: int = 8,
    ) -> None:
        """Load and commit bricks in fill_plan.

        Public name preserved for backwards compatibility.
        Delegates to ``_commit_blocks_async``.

        Parameters
        ----------
        fill_plan : list[tuple[BlockKey, TileSlot]]
            Ordered nearest-first.  Produced by ``plan_update()``.
        status_callback : callable or None
            Optional ``f(text: str) -> None`` called after each batch.
        batch_size : int
            Number of bricks to read and commit before yielding to Qt.
        """
        await self._commit_blocks_async(
            fill_plan,
            status_callback=status_callback,
            batch_size=batch_size,
            label="bricks",
        )


# Deprecated alias — remove in a future cleanup pass.
BlockVolumeState = BlockState3D
