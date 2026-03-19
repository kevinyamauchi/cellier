"""LUT-based brick-cache volume renderer for pygfx.

Phase 3: async brick loading via tensorstore + PySide6.QtAsyncio.

Changes from Phase 2
--------------------
- ``BlockVolumeState`` replaces ``levels: list[np.ndarray]`` with
  ``ts_stores: list[ts.TensorStore]`` (opened synchronously at startup).
- ``BlockVolumeState.update()`` is split into:
    - ``plan_update()``          — synchronous, identical planning logic
    - ``commit_bricks_async()``  — async coroutine, one ``await`` per brick
- ``_read_brick()`` is replaced by ``_read_brick_async()``.
- LOD selection and distance sort are now vectorised (see ``lod.py``).
- The frustum corner construction is vectorised (see ``frustum.py``).

Invariants preserved from Phase 2
----------------------------------
1. Thresholds are screen-space, captured before task submission.
2. Required set sorted nearest-first before staging.
3. Required set hard-capped at ``n_slots`` before staging.
4. Frustum culling after distance sort, before truncation.
5. All pygfx scene graph mutations on the Qt main thread (QtAsyncio).
6. LUT rebuild is O(n_resident_bricks) — now called once per brick.
7. ``update_mode="continuous"`` drives GPU uploads.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pygfx as gfx
import tensorstore as ts

from block_volume.cache import (
    CacheInfo,
    build_cache_texture,
    commit_brick,
    compute_cache_info,
)
from block_volume.frustum import bricks_in_frustum_arr
from block_volume.layout import BLOCK_SIZE_DEFAULT, BlockLayout
from block_volume.lod import (
    build_level_grids,
    select_levels_from_cache,
    select_levels_arr_forced,
    sort_arr_by_distance,
    arr_to_brick_keys,
)
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
    caller can trigger explicit LOD + cache updates.

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
        Number of LOD levels.
    frame_number : int
        Monotonically increasing update counter.
    _level_grids : list[dict]
        Precomputed per-level coarse grid arrays (static after init).
        See ``lod.build_level_grids`` for structure.
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
        # Precompute static coarse grid arrays for each LOD level.
        # Runs once at startup; plan_update reads the cache, no allocation.
        self._level_grids = build_level_grids(layouts[0], self.n_levels)
        total_bricks = sum(len(g["arr"]) for g in self._level_grids)
        print(f"  LOD grid cache: {self.n_levels} levels, "
              f"{total_bricks} total coarse bricks cached")

    # ------------------------------------------------------------------
    # Synchronous planning phase (identical logic to Phase 2 update())
    # ------------------------------------------------------------------

    def plan_update(
        self,
        camera_pos: np.ndarray,
        thresholds: list[float] | None = None,
        force_level: int | None = None,
        frustum_planes: np.ndarray | None = None,
    ) -> tuple[list[tuple[BrickKey, object]], dict]:
        """Run the synchronous planning phase.

        Executes: vectorised LOD select → vectorised distance sort →
        frustum cull → budget truncation → stage().

        Does **not** touch the cache texture or LUT.

        Parameters
        ----------
        camera_pos : np.ndarray
            Camera world-space position ``(x, y, z)``.
        thresholds : list[float] or None
            Screen-space LOD distance cutoffs.
        force_level : int or None
            Override all bricks to this LOD level.
        frustum_planes : ndarray, shape (6, 4) or None
            Inward-pointing frustum half-space planes.

        Returns
        -------
        fill_plan : list[tuple[BrickKey, TileSlot]]
            Ordered nearest-first.  Empty if all bricks are cache hits.
        stats : dict
            Timing and diagnostic information.
        """
        t_plan_start = time.perf_counter()
        self.frame_number += 1

        # ── 1. LOD selection — reads precomputed cache, no allocation ──
        # select_levels_from_cache enumerates each level's coarse grid
        # directly, applies its distance band, and concatenates.  No
        # deduplication needed (bands are disjoint) and no new arrays
        # are allocated — only distance computation + boolean masking.
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
        lod_select_ms = (time.perf_counter() - t0) * 1000

        # ── 2. Distance sort — uses precomputed centres from cache ─────
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
        print(f"chunks: {n_needed} needed / {n_budget} cache slots  ({n_dropped} dropped)")

        # ── 5. Convert to BrickKey dict — first and only Python loop ──
        # By this point M is small (culled + truncated), so the loop is cheap.
        t0 = time.perf_counter()
        sorted_required = arr_to_brick_keys(brick_arr)

        # ── 6. Stage (find hits and misses, plan fills) ───────────────
        fill_plan = self.tile_manager.stage(sorted_required, self.frame_number)
        stage_ms = (time.perf_counter() - t0) * 1000

        plan_total_ms = (time.perf_counter() - t_plan_start) * 1000

        # ── Fill-plan order debug (first 10 misses) ───────────────────
        # Prints brick key + distance so the caller can verify the commit
        # loop processes bricks nearest-first.
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
                print(f"    [{i:3d}] L{bk.level} ({bk.gz:3d},{bk.gy:3d},{bk.gx:3d})"
                      f"  dist={dist:8.1f}")

        # LUT level breakdown (from current resident state, not this frame's
        # plan — gives a snapshot of what is currently visible).
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
            "lod_select_ms": lod_select_ms,
            "distance_sort_ms": distance_sort_ms,
            "frustum_cull_ms": frustum_cull_ms,
            "stage_ms": stage_ms,
            "plan_total_ms": plan_total_ms,
        }
        return fill_plan, stats

    # ------------------------------------------------------------------
    # Async I/O helpers
    # ------------------------------------------------------------------

    async def _read_brick_async(self, key: BrickKey) -> np.ndarray:
        """Read one padded brick from the tensorstore zarr store.

        The ``await`` suspends the coroutine and yields to the Qt event
        loop while tensorstore fetches the chunk(s) from disk.

        Boundary bricks (where the padded region extends outside the
        volume) are handled by allocating a zeroed output of the full
        padded size and copying the valid read region into the correct
        destination offset — identical to the synchronous ``_read_brick``
        logic from Phase 2.

        Parameters
        ----------
        key : BrickKey
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
        sz0 = max(z0, 0);  sz1 = min(z0 + padded, sd)
        sy0 = max(y0, 0);  sy1 = min(y0 + padded, sh)
        sx0 = max(x0, 0);  sx1 = min(x0 + padded, sw)

        if sz1 > sz0 and sy1 > sy0 and sx1 > sx0:
            # Destination offsets into the brick array.
            dz0 = sz0 - z0;  dz1 = dz0 + (sz1 - sz0)
            dy0 = sy0 - y0;  dy1 = dy0 + (sy1 - sy0)
            dx0 = sx0 - x0;  dx1 = dx0 + (sx1 - sx0)

            # Genuine async read — yields to Qt while I/O is in flight.
            region = np.asarray(
                await store[sz0:sz1, sy0:sy1, sx0:sx1].read(),
                dtype=np.float32,
            )
            brick[dz0:dz1, dy0:dy1, dx0:dx1] = region

        return brick

    # ------------------------------------------------------------------
    # Async commit loop
    # ------------------------------------------------------------------

    async def commit_bricks_async(
        self,
        fill_plan: list[tuple[BrickKey, object]],
        status_callback: "Callable[[str], None] | None" = None,
        batch_size: int = 8,
    ) -> None:
        """Load and commit bricks in fill_plan, batching GPU work for throughput.

        Per-batch behaviour
        -------------------
        1. All ``batch_size`` reads are issued concurrently with
           ``asyncio.gather`` so tensorstore can pipeline I/O.
        2. Each arrived brick is written into the CPU cache array and its
           ``update_range`` is scheduled (cheap — no GPU round-trip yet).
        3. ``rebuild_lut`` is called **once per batch** rather than once per
           brick, reducing LUT texture uploads by ``batch_size``×.
        4. ``await asyncio.sleep(0)`` yields to Qt **once per batch** so the
           renderer flushes all pending ``update_range`` calls and redraws.
           This reduces raycaster invocations from ``total`` to
           ``ceil(total / batch_size)`` — typically 10–20× fewer frames.

        Parameters
        ----------
        fill_plan : list[tuple[BrickKey, TileSlot]]
            Ordered nearest-first.  Produced by ``plan_update()``.
        status_callback : callable or None
            Optional ``f(text: str) -> None`` called after each batch.
            Runs on the Qt main thread (QtAsyncio single-thread model).
        batch_size : int
            Number of bricks to read and commit before yielding to Qt.
            Higher values → fewer render interruptions → faster total load,
            but less visual feedback.  Default 8 is a good balance.
        """
        arrived = 0
        total = len(fill_plan)

        # Split fill_plan into batches.
        batches = [fill_plan[i : i + batch_size] for i in range(0, total, batch_size)]

        try:
            for batch in batches:
                # ── Step 1: issue all reads in this batch concurrently ────
                # asyncio.gather suspends until every read completes, allowing
                # tensorstore to pipeline chunk fetches across the batch.
                results = await asyncio.gather(
                    *[self._read_brick_async(bk) for bk, _slot in batch]
                )

                # ── Step 2: commit each brick into the CPU cache ──────────
                # update_range calls accumulate; they are not flushed to the
                # GPU until Qt renders the next frame (after the yield below).
                for (brick_key, slot), data in zip(batch, results):
                    commit_brick(
                        self.cache_data,
                        self.cache_tex,
                        slot.grid_pos,
                        self.cache_info.padded_block_size,
                        data,
                    )
                    arrived += 1

                # ── Step 3: rebuild LUT once for the whole batch ──────────
                rebuild_lut(
                    self.base_layout,
                    self.tile_manager,
                    self.n_levels,
                    self.lut_data,
                    self.lut_tex,
                )

                # ── Step 4: status update ─────────────────────────────────
                if status_callback is not None:
                    status_callback(f"Loading: {arrived} / {total} bricks")

                # ── Step 5: yield to Qt ───────────────────────────────────
                # The renderer will flush all pending update_range calls and
                # run one raycaster pass showing all bricks committed so far.
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            print(f"  commit cancelled after {arrived}/{total} bricks")
            if status_callback is not None:
                status_callback(f"Cancelled ({arrived}/{total} bricks)")
            raise  # mandatory — re-raise so asyncio marks the task cancelled

        print(f"  commit complete: {arrived}/{total} bricks  "
              f"({len(batches)} batches of up to {batch_size})")
        if status_callback is not None:
            status_callback(f"Ready  ({arrived} bricks loaded)")


# ---------------------------------------------------------------------------
# Store initialisation
# ---------------------------------------------------------------------------


def _detect_zarr_driver(level_path: "pathlib.Path") -> str:
    """Return the correct tensorstore driver for a zarr level directory.

    Detects format by the metadata sentinel file:
    - ``.zarray``   → zarr v2  → driver ``"zarr"``
    - ``zarr.json`` → zarr v3  → driver ``"zarr3"``

    Raises ``FileNotFoundError`` if neither file is present.
    """
    import pathlib as _pathlib
    p = _pathlib.Path(level_path)
    if (p / ".zarray").exists():
        return "zarr"
    if (p / "zarr.json").exists():
        return "zarr3"
    raise FileNotFoundError(
        f"Cannot determine zarr format for '{p}': "
        f"neither '.zarray' (zarr v2) nor 'zarr.json' (zarr v3) found.\n"
        f"Re-run with --make-files to regenerate the store."
    )


def _open_ts_stores(
    zarr_path: "pathlib.Path",
    scale_names: list[str],
) -> list[ts.TensorStore]:
    """Open one tensorstore per scale level (read-only, synchronous).

    Auto-detects zarr v2 vs v3 format from the metadata sentinel file.
    Must be called **before** ``QtAsyncio.run()`` starts the event loop.

    Parameters
    ----------
    zarr_path : pathlib.Path
        Root directory of the multiscale zarr store.
    scale_names : list[str]
        Subdirectory names in order finest → coarsest, e.g.
        ``["s0", "s1", "s2"]``.

    Returns
    -------
    stores : list[ts.TensorStore]
        One open store per scale level.  Chunk data is not loaded yet.
    """
    import pathlib as _pathlib
    stores = []
    for name in scale_names:
        level_path = _pathlib.Path(zarr_path) / name
        driver = _detect_zarr_driver(level_path)
        spec = {
            "driver": driver,
            "kvstore": {
                "driver": "file",
                "path": str(level_path),
            },
        }
        store = ts.open(spec).result()
        print(f"  {name}: opened as zarr {'v2' if driver == 'zarr' else 'v3'}"
              f"  shape={tuple(int(d) for d in store.domain.shape)}")
        stores.append(store)
    return stores


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
) -> tuple[gfx.Volume, BlockVolumeState]:
    """Create a pygfx Volume using multi-LOD LUT-based async rendering.

    Parameters
    ----------
    ts_stores : list[ts.TensorStore]
        Pre-opened tensorstore handles ``[finest, ..., coarsest]``.
        Use :func:`_open_ts_stores` to create them before calling this.
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
    state : BlockVolumeState
        Mutable state object.  Call ``state.plan_update(...)`` +
        ``asyncio.ensure_future(state.commit_bricks_async(...))``
        to drive LOD selection and async cache fills.
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
    print(f"  make_block_volume: layouts          {(time.perf_counter()-t0)*1000:.1f} ms")

    t1 = time.perf_counter()
    cache_info = compute_cache_info(block_size, gpu_budget_bytes)
    cache_data, cache_tex = build_cache_texture(cache_info)
    print(f"  make_block_volume: cache texture    {(time.perf_counter()-t1)*1000:.1f} ms"
          f"  shape={cache_info.cache_shape}")

    t1 = time.perf_counter()
    lut_data, lut_tex = build_lut_texture(base_layout)
    print(f"  make_block_volume: LUT texture      {(time.perf_counter()-t1)*1000:.1f} ms"
          f"  shape={base_layout.grid_dims}")

    tile_manager = TileManager(cache_info)

    lut_params = build_lut_params_buffer(base_layout, cache_info,
                                        proxy_voxels_per_brick=block_size)
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

    # Proxy texture: one voxel per brick (shape = grid_dims), scaled up to
    # the full volume world size via the Volume's local transform.
    #
    # Previously this used base_layout.padded_shape (e.g. 1024³ for a 1024³
    # volume at block_size=32), which allocates a 4 GiB float32 array and
    # uploads it to the GPU — causing multi-second startup delay.
    # Using the grid dims (e.g. 32³ = 32 KB) is ~32000× smaller and the
    # shader never samples it; only textureDimensions(t_img) is read, which
    # returns grid_dims here.  We compensate by scaling the Volume's local
    # transform so the geometry bounding box still covers the correct world
    # extent (block_size voxels per unit).
    t1 = time.perf_counter()
    gd, gh, gw = base_layout.grid_dims
    proxy_data = np.zeros((gd, gh, gw), dtype=np.float32)
    proxy_tex = gfx.Texture(proxy_data, dim=3)
    print(f"  make_block_volume: proxy texture    {(time.perf_counter()-t1)*1000:.1f} ms"
          f"  shape=({gd},{gh},{gw})  [grid dims, not voxel dims]")

    t1 = time.perf_counter()
    geometry = gfx.Geometry(grid=proxy_tex)
    vol = gfx.Volume(geometry, material)

    # Scale so each grid cell covers block_size world units, making the
    # bounding box match the true voxel extent of the volume.
    vol.local.scale = (float(block_size), float(block_size), float(block_size))
    # Correct the origin offset introduced by the scale.
    # With scale=bs, data coord -0.5 → world -0.5*bs.
    # We want data -0.5 → world -0.5 (same as the original unscaled proxy).
    # Translation needed: offset = -0.5 - (-0.5*bs) = 0.5*(bs-1)
    _off = 0.5 * (float(block_size) - 1.0)
    vol.local.position = (_off, _off, _off)
    print(f"  make_block_volume: volume object    {(time.perf_counter()-t1)*1000:.1f} ms")
    print(f"  make_block_volume: TOTAL            {(time.perf_counter()-t0)*1000:.1f} ms")

    state = BlockVolumeState(
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
