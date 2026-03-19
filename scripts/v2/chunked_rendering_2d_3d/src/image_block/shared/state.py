"""Block state for 2D tiled image rendering with async loading.

Contains the synchronous planning phase (``plan_update``) and the
async commit loop (``commit_tiles_async``) that loads tile data from
tensorstore and uploads to the GPU cache.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Callable

import numpy as np
import pygfx as gfx
import tensorstore as ts

from image_block.shared.cache import CacheInfo, commit_tile
from image_block.shared.culling import viewport_cull
from image_block.shared.layout import BlockLayout2D
from image_block.shared.lod import (
    arr_to_tile_keys,
    build_tile_grids,
    select_lod_2d,
    sort_tiles_by_distance,
)
from image_block.shared.lut import rebuild_lut
from image_block.shared.tile_manager import TileKey, TileManager, TileSlot


class BlockState2D:
    """Mutable state for a multi-LOD 2D tiled image.

    Created by :func:`make_block_image` and exposed so the caller can
    trigger LOD + cache updates.

    Attributes
    ----------
    ts_stores : list[ts.TensorStore]
        Open tensorstore handles ``[finest, ..., coarsest]``.
    layouts : list[BlockLayout2D]
        One layout per level.
    base_layout : BlockLayout2D
        Layout of the finest level.
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
        Tile-to-slot mapping with LRU eviction.
    block_size : int
        Tile side length in pixels.
    overlap : int
        Border pixels per side.
    n_levels : int
        Number of LOD levels.
    z_slice : int
        Which Z slice to extract from 3D stores.
    frame_number : int
        Monotonically increasing update counter.
    """

    def __init__(
        self,
        ts_stores: list[ts.TensorStore],
        layouts: list[BlockLayout2D],
        cache_info: CacheInfo,
        cache_data: np.ndarray,
        cache_tex: gfx.Texture,
        lut_data: np.ndarray,
        lut_tex: gfx.Texture,
        tile_manager: TileManager,
        block_size: int,
        overlap: int = 1,
        z_slice: int | None = None,
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

        # Determine z_slice per level.
        if z_slice is None:
            # Default: mid-slice of the finest level.
            z_shape = int(ts_stores[0].domain.shape[0])
            z_slice = z_shape // 2
        self.z_slice = z_slice

        # Precompute per-level z-slice indices.
        self._z_slices: list[int] = []
        for i in range(self.n_levels):
            scale = 2 ** i
            self._z_slices.append(z_slice // scale)

        # Precompute static tile grids.
        self._level_grids = build_tile_grids(self.base_layout, self.n_levels)
        total_tiles = sum(len(g["arr"]) for g in self._level_grids)
        print(
            f"  LOD grid cache: {self.n_levels} levels, "
            f"{total_tiles} total coarse tiles cached"
        )

    # ------------------------------------------------------------------
    # Synchronous planning phase
    # ------------------------------------------------------------------

    def plan_update(
        self,
        camera_info: dict,
        lod_bias: float = 1.0,
        force_level: int | None = None,
        use_culling: bool = True,
    ) -> tuple[list[tuple[TileKey, TileSlot]], dict]:
        """Run the synchronous planning phase.

        Executes: LOD select -> distance sort -> viewport cull ->
        budget truncation -> stage.

        Parameters
        ----------
        camera_info : dict
            Camera state snapshot.
        lod_bias : float
            Scale factor for LOD thresholds.
        force_level : int or None
            Force a specific LOD level.
        use_culling : bool
            Enable viewport culling.

        Returns
        -------
        fill_plan : list[tuple[TileKey, TileSlot]]
            Tiles needing data upload.
        stats : dict
            Planning statistics and timing.
        """
        self.frame_number += 1
        stats: dict = {}
        t0 = time.perf_counter()

        # 1. LOD selection.
        t_lod = time.perf_counter()
        arr = select_lod_2d(
            self._level_grids,
            self.n_levels,
            camera_info,
            lod_bias=lod_bias,
            force_level=force_level,
        )
        stats["lod_select_ms"] = (time.perf_counter() - t_lod) * 1000

        # Collect level counts.
        level_counts: dict[int, int] = {}
        if len(arr) > 0:
            unique_levels, counts = np.unique(arr[:, 0], return_counts=True)
            for lv, ct in zip(unique_levels, counts):
                level_counts[int(lv)] = int(ct)
        stats["level_counts"] = level_counts

        # 2. Distance sort (nearest to view centre first).
        t_sort = time.perf_counter()
        arr = sort_tiles_by_distance(arr, camera_info, self.block_size)
        stats["distance_sort_ms"] = (time.perf_counter() - t_sort) * 1000

        # Convert to dict.
        required = arr_to_tile_keys(arr)
        stats["total_required"] = len(required)

        # 3. Viewport culling.
        n_culled = 0
        if use_culling and "view_min" in camera_info:
            t_cull = time.perf_counter()
            required, n_culled = viewport_cull(
                required, self.base_layout, camera_info
            )
            stats["cull_ms"] = (time.perf_counter() - t_cull) * 1000
        stats["n_culled"] = n_culled

        # 4. Budget truncation: hard-cap at n_slots.
        n_budget = self.cache_info.n_slots
        n_needed = len(required)
        n_dropped = max(0, n_needed - n_budget)
        if n_dropped > 0:
            # Keep only the first n_budget entries (nearest-first).
            keys_to_keep = list(required.keys())[:n_budget]
            required = {k: required[k] for k in keys_to_keep}
        stats["n_needed"] = n_needed
        stats["n_budget"] = n_budget
        stats["n_dropped"] = n_dropped

        # 5. Stage: identify hits and misses.
        t_stage = time.perf_counter()
        fill_plan = self.tile_manager.stage(required, self.frame_number)
        stats["stage_ms"] = (time.perf_counter() - t_stage) * 1000

        hits = len(required) - len(fill_plan)
        stats["hits"] = hits
        stats["fills"] = len(fill_plan)

        stats["plan_total_ms"] = (time.perf_counter() - t0) * 1000
        return fill_plan, stats

    # ------------------------------------------------------------------
    # Async commit loop
    # ------------------------------------------------------------------

    async def _read_tile_async(self, key: TileKey) -> np.ndarray:
        """Read one padded 2D tile from the tensorstore zarr store.

        Reads ``store[z_mid, y0:y1, x0:x1]`` (or ``store[y0:y1, x0:x1]``
        for 2D stores) where the indices include the overlap border.
        Pads with zeros for boundary tiles that extend outside the array.

        Parameters
        ----------
        key : TileKey
            Tile to read.

        Returns
        -------
        data : np.ndarray
            Float32 array of shape ``(pbs, pbs)``.
        """
        level_idx = key.level - 1
        store = self.ts_stores[level_idx]
        layout = self.layouts[level_idx]

        bs = self.block_size
        ov = self.overlap
        pbs = bs + 2 * ov

        # Tile origin in data coordinates (no overlap).
        y_start = key.gy * bs
        x_start = key.gx * bs

        # Padded region including overlap.
        y0 = y_start - ov
        x0 = x_start - ov
        y1 = y0 + pbs
        x1 = x0 + pbs

        h, w = layout.volume_shape

        # Clamp to valid range.
        cy0 = max(y0, 0)
        cx0 = max(x0, 0)
        cy1 = min(y1, h)
        cx1 = min(x1, w)

        # Allocate output (zero-padded for boundary tiles).
        out = np.zeros((pbs, pbs), dtype=np.float32)

        if cy0 >= cy1 or cx0 >= cx1:
            return out

        # Offsets into the output array.
        oy0 = cy0 - y0
        ox0 = cx0 - x0
        oy1 = oy0 + (cy1 - cy0)
        ox1 = ox0 + (cx1 - cx0)

        # Read from tensorstore.  The store may be 3D (Z, Y, X) or 2D (Y, X).
        ndim = len(store.domain.shape)
        if ndim == 3:
            z_mid = self._z_slices[level_idx]
            chunk = await store[z_mid, cy0:cy1, cx0:cx1].read()
        elif ndim == 2:
            chunk = await store[cy0:cy1, cx0:cx1].read()
        else:
            raise ValueError(f"Unexpected store ndim={ndim}")

        out[oy0:oy1, ox0:ox1] = np.asarray(chunk, dtype=np.float32)
        return out

    async def commit_tiles_async(
        self,
        fill_plan: list[tuple[TileKey, TileSlot]],
        status_callback: Callable[[str], None] | None = None,
        batch_size: int = 8,
    ) -> None:
        """Batched async commit loop.

        Per batch:
        1. ``asyncio.gather`` all reads in the batch (concurrent I/O).
        2. Commit each tile into the CPU cache array + ``update_range``.
        3. ``rebuild_lut()`` once for the whole batch.
        4. ``await asyncio.sleep(0)`` to yield to Qt (GPU flush + redraw).

        Re-raises ``CancelledError`` so asyncio marks the task cancelled.

        Parameters
        ----------
        fill_plan : list[tuple[TileKey, TileSlot]]
            Produced by ``plan_update()``.
        status_callback : callable or None
            Optional ``f(text) -> None`` for status updates.
        batch_size : int
            Tiles per batch before yielding.
        """
        arrived = 0
        total = len(fill_plan)

        batches = [
            fill_plan[i : i + batch_size]
            for i in range(0, total, batch_size)
        ]

        try:
            for batch in batches:
                # Step 1: concurrent reads.
                results = await asyncio.gather(
                    *[self._read_tile_async(tk) for tk, _slot in batch]
                )

                # Step 2: commit each tile into the CPU cache.
                for (tile_key, slot), data in zip(batch, results):
                    commit_tile(
                        self.cache_data,
                        self.cache_tex,
                        slot.grid_pos,
                        self.cache_info.padded_block_size,
                        data,
                    )
                    arrived += 1

                # Step 3: rebuild LUT once for the whole batch.
                rebuild_lut(
                    self.base_layout,
                    self.tile_manager,
                    self.n_levels,
                    self.lut_data,
                    self.lut_tex,
                )

                # Step 4: status update.
                if status_callback is not None:
                    status_callback(f"Loading: {arrived} / {total} tiles")

                # Step 5: yield to Qt.
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            print(f"  commit cancelled after {arrived}/{total} tiles")
            if status_callback is not None:
                status_callback(f"Cancelled ({arrived}/{total} tiles)")
            raise  # mandatory -- re-raise so asyncio marks the task cancelled

        print(
            f"  commit complete: {arrived}/{total} tiles  "
            f"({len(batches)} batches of up to {batch_size})"
        )
        if status_callback is not None:
            status_callback(f"Ready  ({arrived} tiles loaded)")
