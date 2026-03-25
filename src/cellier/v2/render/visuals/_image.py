"""GFXMultiscaleImageVisual — 3D image visual using LUT brick-cache rendering."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx

from cellier.v2.data.image import ChunkRequest
from cellier.v2.render._frustum import bricks_in_frustum_arr
from cellier.v2.render._level_of_detail import (
    arr_to_brick_keys,
    build_level_grids,
    select_levels_arr_forced,
    select_levels_from_cache,
    sort_arr_by_distance,
)
from cellier.v2.render.block_cache import (
    BlockCache3D,
    BlockKey3D,
    TileSlot,
    compute_block_cache_parameters_3d,
)
from cellier.v2.render.block_cache._block_cache_2d import BlockCache2D
from cellier.v2.render.block_cache._cache_parameters_2d import (
    compute_block_cache_parameters_2d,
)
from cellier.v2.render.lut_indirection import BlockLayout3D, LutIndirectionManager3D
from cellier.v2.render.lut_indirection._layout_2d import BlockLayout2D
from cellier.v2.render.lut_indirection._lut_buffers_2d import (
    build_block_scales_buffer_2d,
    build_lut_params_buffer_2d,
)
from cellier.v2.render.lut_indirection._lut_indirection_manager_2d import (
    LutIndirectionManager2D,
)
from cellier.v2.render.shaders._block_image import ImageBlockMaterial
from cellier.v2.render.shaders._block_volume import (
    VolumeBlockMaterial,
    build_block_scales_buffer_3d,
    build_lut_params_buffer_3d,
)

if TYPE_CHECKING:
    from cellier.v2.events._events import (
        AppearanceChangedEvent,
        DataStoreContentsChangedEvent,
        DataStoreMetadataChangedEvent,
        VisualVisibilityChangedEvent,
    )
    from cellier.v2.visuals._image import MultiscaleImageVisual

# Importing this module registers VolumeBlockShader with pygfx via the
# @register_wgpu_render_function decorator.
import cellier.v2.render.shaders._block_volume as _shader_reg  # noqa: F401

# ---------------------------------------------------------------------------
# VolumeGeometry
# ---------------------------------------------------------------------------


class VolumeGeometry:
    """Pre-built metadata cache for a multiscale 3-D volume.

    Holds per-level ``BlockLayout3D`` objects and the precomputed coarse
    grid arrays used by the LOAD selection pipeline.  Never touches the
    live ``DataStore`` after construction.

    Parameters
    ----------
    level_shapes : list[tuple[int, int, int]]
        Volume shape ``(D, H, W)`` at each scale level, finest first.
    downscale_factors : list[int]
        Downscale factor for each level relative to the original data,
        e.g. ``[1, 2, 4]``.  Must have the same length as
        ``level_shapes``.
    block_size : int
        Rendering brick side length in voxels.
    """

    def __init__(
        self,
        level_shapes: list[tuple[int, int, int]],
        downscale_factors: list[int],
        block_size: int,
    ) -> None:
        self.downscale_factors = list(downscale_factors)
        self.block_size = block_size
        self.n_levels = len(level_shapes)
        self._rebuild(level_shapes)

    @classmethod
    def from_cellier_model(
        cls,
        model: MultiscaleImageVisual,
        level_shapes: list[tuple[int, int, int]],
        block_size: int,
    ) -> VolumeGeometry:
        """Build a ``VolumeGeometry`` from a ``MultiscaleImageVisual`` model.

        Parameters
        ----------
        model : MultiscaleImageVisual
            The visual model.  Only ``downscale_factors`` is read.
        level_shapes : list[tuple[int, int, int]]
            Volume shape ``(D, H, W)`` per level, finest first.  Must be
            obtained from the data store by the caller.
        block_size : int
            Rendering brick side length in voxels.

        Returns
        -------
        VolumeGeometry
        """
        return cls(
            level_shapes=level_shapes,
            downscale_factors=model.downscale_factors,
            block_size=block_size,
        )

    def _rebuild(self, level_shapes: list[tuple[int, int, int]]) -> None:
        self.level_shapes = list(level_shapes)
        self.layouts = [
            BlockLayout3D(volume_shape=shape, block_size=self.block_size)
            for shape in level_shapes
        ]
        self.base_layout = self.layouts[0]
        self._level_grids = build_level_grids(self.base_layout, self.n_levels)

    def update(self, level_shapes: list[tuple[int, int, int]]) -> None:
        """Rebuild from new level shapes after a DataStoreMutated event."""
        self._rebuild(level_shapes)


# ---------------------------------------------------------------------------
# Coordinate helper
# ---------------------------------------------------------------------------


def _brick_key_to_padded_coords(
    key: BlockKey3D,
    block_size: int,
    overlap: int,
) -> tuple[int, int, int, int, int, int]:
    """Translate a BlockKey3D to padded voxel coordinates.

    Returns z0, y0, x0, z1, y1, x1.  May extend outside store bounds;
    the DataStore is responsible for clamping and zero-padding.
    """
    padded = block_size + 2 * overlap
    z0 = key.gz * block_size - overlap
    y0 = key.gy * block_size - overlap
    x0 = key.gx * block_size - overlap
    return z0, y0, x0, z0 + padded, y0 + padded, x0 + padded


# ---------------------------------------------------------------------------
# GFXMultiscaleImageVisual
# ---------------------------------------------------------------------------


class GFXMultiscaleImageVisual:
    """Render-layer wrapper for one logical multiscale image visual.

    Owns the 3D pygfx ``Volume`` node, the GPU brick cache, and the LUT
    indirection texture.  Implements ``build_slice_request()`` and
    ``on_data_ready()`` for use by a ``SliceCoordinator`` (or, in the
    prototype, directly by the application).

    The associated ``MultiscaleImageVisual`` model is identified only by
    ``visual_model_id`` — no model reference crosses the render-layer
    boundary.

    Parameters
    ----------
    visual_model_id : UUID
        ID of the associated ``MultiscaleImageVisual`` model.
    volume_geometry : VolumeGeometry
        Pre-built metadata cache (level shapes, LOAD grids).
    render_modes : set[str]
        Which nodes to build: ``{"3d"}``, ``{"2d"}``, or
        ``{"2d", "3d"}``.  Non-applicable nodes are ``None``.
    colormap : gfx.TextureMap or None
        Colourmap for the volume.  Defaults to viridis.
    clim : tuple[float, float]
        Contrast limits passed to the material.
    threshold : float
        Isosurface threshold.
    interpolation : str
        Sampler filter (``"linear"`` or ``"nearest"``).
    gpu_budget_bytes : int
        Maximum GPU memory for the brick cache texture.
    """

    def __init__(
        self,
        visual_model_id: UUID,
        volume_geometry: VolumeGeometry,
        render_modes: set[str],
        colormap: gfx.TextureMap | None = None,
        clim: tuple[float, float] = (0.0, 1.0),
        threshold: float = 0.5,
        interpolation: str = "linear",
        gpu_budget_bytes: int = 1 * 1024**3,
    ) -> None:
        self.visual_model_id = visual_model_id
        self.render_modes = render_modes
        self._volume_geometry = volume_geometry
        self._frame_number = 0
        self._pending_slot_map: dict[UUID, tuple[BlockKey3D, TileSlot]] = {}
        self._last_plan_stats: dict = {}

        # ── GPU resources ───────────────────────────────────────────────
        cache_parameters_3d = compute_block_cache_parameters_3d(
            block_size=volume_geometry.block_size,
            gpu_budget_bytes=gpu_budget_bytes,
        )
        self._block_cache_3d = BlockCache3D(cache_parameters=cache_parameters_3d)
        self._lut_manager_3d = LutIndirectionManager3D(
            base_layout=volume_geometry.base_layout,
            n_levels=volume_geometry.n_levels,
        )
        self._lut_params_buffer_3d = build_lut_params_buffer_3d(
            volume_geometry.base_layout,
            cache_parameters_3d,
            proxy_voxels_per_brick=volume_geometry.block_size,
        )
        self._block_scales_buffer_3d = build_block_scales_buffer_3d(
            volume_geometry.downscale_factors
        )

        # 2D GPU resources
        # 2. Compute cache info.
        cache_parameters_2d = compute_block_cache_parameters_2d(
            gpu_budget_bytes=gpu_budget_bytes, block_size=volume_geometry.block_size
        )
        self._block_cache_2d = BlockCache2D(cache_parameters=cache_parameters_2d)
        pbs = cache_parameters_2d.padded_block_size
        gs = cache_parameters_2d.grid_side
        cache_pixels = gs * pbs
        cache_mb = cache_pixels * cache_pixels * 4 / (1024**2)
        print(
            f"  cache: {gs}x{gs} slots = {cache_parameters_2d.n_slots} slots, "
            f"texture {cache_pixels}x{cache_pixels} ({cache_mb:.1f} MB)"
        )

        # 3. Build LUT texture.
        # need to integrate base_layout with VolumeGeometry
        base_layout = BlockLayout2D.from_shape(
            shape=volume_geometry.level_shapes[0][1:3],
            block_size=volume_geometry.block_size,
        )
        self._lut_manager_2d = LutIndirectionManager2D(
            base_layout=base_layout,
            n_levels=volume_geometry.n_levels,
        )

        # 4. Build uniform buffers.
        self._lut_params_buffer_2d = build_lut_params_buffer_2d(
            base_layout, cache_parameters_2d
        )
        self._block_scales_buffer_2d = build_block_scales_buffer_2d(
            volume_geometry.n_levels
        )

        if colormap is None:
            colormap = gfx.cm.viridis

        # ── 3D node ─────────────────────────────────────────────────────
        self.node_3d: gfx.Volume | None = None
        self.material_3d: VolumeBlockMaterial | None = None
        self._proxy_tex_3d: gfx.Texture | None = None
        if "3d" in render_modes:
            self.node_3d, self.material_3d, self._proxy_tex_3d = self._build_3d_node(
                colormap=colormap,
                clim=clim,
                threshold=threshold,
                interpolation=interpolation,
            )

        # ── 2D node -----------------------───────────────────────────────
        self.node_2d = gfx.Image | None
        self.material_2d = ImageBlockMaterial | None
        if "2d" in render_modes:
            print(colormap)
            self.node_2d, self.material_2d, self._proxy_tex_2d = self._build_2d_node(
                colormap=colormap, clim=clim, interpolation=interpolation
            )

    @classmethod
    def from_cellier_model(
        cls,
        model: MultiscaleImageVisual,
        level_shapes: list[tuple[int, int, int]],
        render_modes: set[str],
        block_size: int = 32,
        gpu_budget_bytes: int = 1 * 1024**3,
        threshold: float = 0.5,
        interpolation: str = "linear",
    ) -> GFXMultiscaleImageVisual:
        """Build a ``GFXMultiscaleImageVisual`` from a ``MultiscaleImageVisual`` model.

        Reads appearance parameters from the model and constructs the visual
        with plain values only.  The model reference does not cross the
        render-layer boundary after this call.

        Parameters
        ----------
        model : MultiscaleImageVisual
            Source visual model.  ``model.id``,
            ``model.appearance.color_map``, ``model.appearance.clim``,
            and ``model.downscale_factors`` are consumed.
        level_shapes : list[tuple[int, int, int]]
            Volume shape ``(D, H, W)`` per level, finest first.  Read
            from the data store by the caller before invoking this method.
        render_modes : set[str]
            Which nodes to build: ``{"3d"}``, ``{"2d"}``, or
            ``{"2d", "3d"}``.
        block_size : int
            Rendering brick side length in voxels.
        gpu_budget_bytes : int
            Maximum GPU memory for the brick cache.
        threshold : float
            Isosurface threshold for 3D raycast rendering.
        interpolation : str
            Sampler filter (``"linear"`` or ``"nearest"``).

        Returns
        -------
        GFXMultiscaleImageVisual
        """
        volume_geometry = VolumeGeometry.from_cellier_model(
            model, level_shapes, block_size
        )
        colormap = model.appearance.color_map.to_pygfx(N=256)
        clim = model.appearance.clim

        return cls(
            visual_model_id=model.id,
            volume_geometry=volume_geometry,
            render_modes=render_modes,
            colormap=colormap,
            clim=clim,
            threshold=threshold,
            interpolation=interpolation,
            gpu_budget_bytes=gpu_budget_bytes,
        )

    # ── SliceCoordinator interface ──────────────────────────────────────

    def build_slice_request(
        self,
        camera_pos: np.ndarray,
        frustum_planes: np.ndarray | None,
        thresholds: list[float] | None,
        force_level: int | None = None,
    ) -> list[ChunkRequest]:
        """Run the synchronous planning phase and return ChunkRequests.

        Pipeline: LOAD select → distance sort → optional frustum cull →
        budget cap → stage() → build ``ChunkRequest`` objects.

        Side effects: populates ``self._pending_slot_map`` (replacing any
        previous map) and ``self._last_plan_stats``.

        Parameters
        ----------
        camera_pos : ndarray, shape (3,)
            Camera world-space position ``(x, y, z)``.
        frustum_planes : ndarray, shape (6, 4) or None
            Inward-pointing half-space planes.  ``None`` disables
            frustum culling.
        thresholds : list[float] or None
            LOAD cutoff distances, one per level boundary.
        force_level : int or None
            Override: all bricks assigned this 1-based level.

        Returns
        -------
        list[ChunkRequest]
            Nearest-first.  Empty when all required bricks are cached.
        """
        t_plan_start = time.perf_counter()
        self._frame_number += 1
        geo = self._volume_geometry

        # 1. LOAD selection
        t0 = time.perf_counter()
        if force_level is not None:
            brick_arr = select_levels_arr_forced(
                geo.base_layout, force_level, geo._level_grids
            )
        else:
            brick_arr = select_levels_from_cache(
                geo._level_grids,
                geo.n_levels,
                camera_pos,
                thresholds=thresholds,
                base_layout=geo.base_layout,
            )
        lod_select_ms = (time.perf_counter() - t0) * 1000

        # 2. Distance sort
        t0 = time.perf_counter()
        brick_arr = sort_arr_by_distance(brick_arr, camera_pos, geo.block_size)
        distance_sort_ms = (time.perf_counter() - t0) * 1000

        n_total = len(brick_arr)

        # 3. Frustum cull
        cull_timings: dict = {}
        n_culled = 0
        frustum_cull_ms = 0.0
        if frustum_planes is not None:
            t0 = time.perf_counter()
            brick_arr, cull_timings = bricks_in_frustum_arr(
                brick_arr, geo.block_size, frustum_planes
            )
            frustum_cull_ms = (time.perf_counter() - t0) * 1000
            n_culled = n_total - len(brick_arr)

        # 4. Budget truncation
        n_needed = len(brick_arr)
        n_budget = self._block_cache_3d.info.n_slots - 1
        n_dropped = max(0, n_needed - n_budget)
        if n_dropped:
            brick_arr = brick_arr[:n_budget]

        # 5. Stage: find cache hits/misses, reserve slots for misses
        t0 = time.perf_counter()
        sorted_required = arr_to_brick_keys(brick_arr)
        fill_plan = self._block_cache_3d.tile_manager.stage(
            sorted_required, self._frame_number
        )
        stage_ms = (time.perf_counter() - t0) * 1000

        # 6. Build ChunkRequests and populate the pending slot map
        slice_id = uuid4()
        chunk_requests: list[ChunkRequest] = []
        self._pending_slot_map = {}

        for brick_key, slot in fill_plan:
            chunk_id = uuid4()
            z0, y0, x0, z1, y1, x1 = _brick_key_to_padded_coords(
                brick_key, geo.block_size, self._block_cache_3d.info.overlap
            )
            req = ChunkRequest(
                chunk_request_id=chunk_id,
                slice_request_id=slice_id,
                scale_index=brick_key.level - 1,
                z_start=z0,
                y_start=y0,
                x_start=x0,
                z_stop=z1,
                y_stop=y1,
                x_stop=x1,
            )
            chunk_requests.append(req)
            self._pending_slot_map[chunk_id] = (brick_key, slot)

        plan_total_ms = (time.perf_counter() - t_plan_start) * 1000

        # 7. Snapshot LUT level breakdown for debug reporting
        level_counts: dict[int, int] = {}
        gd, gh, gw = geo.base_layout.grid_dims
        for gz in range(gd):
            for gy in range(gh):
                for gx in range(gw):
                    lv = int(self._lut_manager_3d.lut_data[gz, gy, gx, 3])
                    level_counts[lv] = level_counts.get(lv, 0) + 1

        self._last_plan_stats = {
            "hits": len(sorted_required) - len(fill_plan),
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

        return chunk_requests

    def on_data_ready(
        self,
        batch: list[tuple[ChunkRequest, np.ndarray]],
    ) -> None:
        """Commit an arriving batch of bricks to the GPU cache.

        Writes each brick into the cache texture and advances it from
        in-flight to committed in the tile manager.  Rebuilds the LUT
        once per batch (not once per brick) for efficiency.

        Parameters
        ----------
        batch : list of (ChunkRequest, ndarray)
            Each item is a ``(request, data)`` pair where ``data`` has
            shape ``(padded_D, padded_H, padded_W)``, dtype float32.
        """
        for req, data in batch:
            entry = self._pending_slot_map.get(req.chunk_request_id)
            if entry is None:
                continue
            brick_key, slot = entry
            self._block_cache_3d.write_brick(slot, data)
            self._block_cache_3d.tile_manager.commit(brick_key, slot)

        self._lut_manager_3d.rebuild(self._block_cache_3d.tile_manager)

    def cancel_pending(self) -> None:
        """Release all in-flight slots reserved by the last build_slice_request.

        Call this before submitting a new ``build_slice_request()`` to
        cleanly reclaim slots that were reserved but never committed.
        Only previously committed (valid) bricks remain renderable after
        this call.
        """
        self._block_cache_3d.tile_manager.release_all_in_flight()
        self._pending_slot_map = {}

    # ── EventBus handler methods ─────────────────────────────────────────

    def on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        """Apply GPU-only appearance changes.

        Reslice-field changes (lod_bias, force_level, frustum_cull) are
        no-ops here; new values are read from the model at the next
        trigger_update() call.
        """
        if event.field_name == "color_map":
            from cmap import Colormap

            colormap = (
                Colormap(event.new_value)
                if isinstance(event.new_value, str)
                else event.new_value
            )
            new_map = colormap.to_pygfx(N=256)
            if self.material_3d is not None:
                self.material_3d.map = new_map
            if self.material_2d is not None:
                self.material_2d.map = new_map
        elif event.field_name == "clim":
            if self.material_3d is not None:
                self.material_3d.clim = event.new_value
            if self.material_2d is not None:
                self.material_2d.clim = event.new_value
        # lod_bias / force_level / frustum_cull: no GPU state to update here.

    def on_visibility_changed(self, event: VisualVisibilityChangedEvent) -> None:
        """Apply visibility change to all render nodes."""
        if self.node_3d is not None:
            self.node_3d.visible = event.visible
        if self.node_2d is not None:
            self.node_2d.visible = event.visible

    def on_data_store_contents_changed(
        self, event: DataStoreContentsChangedEvent
    ) -> None:
        """Stub — brick eviction deferred to a future phase."""
        pass

    def on_data_store_metadata_changed(
        self, event: DataStoreMetadataChangedEvent
    ) -> None:
        """Stub — geometry rebuild deferred to a future phase."""
        pass

    # ── Private helpers ─────────────────────────────────────────────────

    def _build_3d_node(
        self,
        colormap: gfx.TextureMap,
        clim: tuple[float, float],
        threshold: float,
        interpolation: str,
    ) -> tuple[gfx.Volume, VolumeBlockMaterial, gfx.Texture]:
        """Construct the proxy texture, material, and Volume node."""
        geo = self._volume_geometry
        gd, gh, gw = geo.base_layout.grid_dims

        proxy_data = np.zeros((gd, gh, gw), dtype=np.float32)
        proxy_tex = gfx.Texture(proxy_data, dim=3)

        material = VolumeBlockMaterial(
            cache_texture=self._block_cache_3d.cache_tex,
            lut_texture=self._lut_manager_3d.lut_tex,
            lut_params_buffer=self._lut_params_buffer_3d,
            block_scales_buffer=self._block_scales_buffer_3d,
            clim=clim,
            map=colormap,
            interpolation=interpolation,
            threshold=threshold,
        )

        geometry = gfx.Geometry(grid=proxy_tex)
        vol = gfx.Volume(geometry, material)

        bs = float(geo.block_size)
        vol.local.scale = (bs, bs, bs)
        off = 0.5 * (bs - 1.0)
        vol.local.position = (off, off, off)

        return vol, material, proxy_tex

    def _build_2d_node(
        self,
        colormap: gfx.TextureMap,
        clim: tuple[float, float],
        interpolation: str,
    ) -> tuple[gfx.Image, ImageBlockMaterial, gfx.Texture]:
        gh, gw = self._lut_manager_2d._base_layout.grid_dims

        proxy_data = np.zeros((gh, gw), dtype=np.float32)
        proxy_tex = gfx.Texture(proxy_data, dim=2)

        material = ImageBlockMaterial(
            cache_texture=self._block_cache_2d.cache_tex,
            lut_texture=self._lut_manager_2d.lut_tex,
            lut_params_buffer=self._lut_params_buffer_2d,
            block_scales_buffer=self._block_scales_buffer_2d,
            clim=clim,
            map=colormap,
        )

        # 8. Build Image.
        geometry = gfx.Geometry(grid=proxy_tex)
        image = gfx.Image(geometry, material)

        # 9. Scale: proxy has 1 texel per tile, so scale by block_size
        #    to make the quad cover the correct world-space extent.
        image.local.scale = (
            self._volume_geometry.block_size,
            self._volume_geometry.block_size,
            1.0,
        )

        # get_im_geometry() positions the quad from (-0.5, -0.5) to
        # (gW-0.5, gH-0.5) in proxy texels.  After scaling by block_size
        # the quad spans (-bs/2, -bs/2) to (gW*bs - bs/2, gH*bs - bs/2).
        # Shift by +bs/2 so it sits at (0, 0) to (gW*bs, gH*bs).
        image.local.position = (
            self._volume_geometry.block_size * 0.5,
            self._volume_geometry.block_size * 0.5,
            0,
        )

        return image, material, proxy_tex
