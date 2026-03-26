"""GFXMultiscaleImageVisual — render-layer visual for multiscale images."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx

from cellier.v2.data.image import ChunkRequest
from cellier.v2.logging import _GPU_LOGGER, _PERF_LOGGER
from cellier.v2.render._frustum import bricks_in_frustum_arr
from cellier.v2.render._level_of_detail import (
    arr_to_brick_keys,
    build_level_grids,
    select_levels_arr_forced,
    select_levels_from_cache,
    sort_arr_by_distance,
)
from cellier.v2.render._level_of_detail_2d import (
    arr_to_block_keys_2d,
    build_tile_grids_2d,
    select_lod_2d,
    sort_tiles_by_distance_2d,
    viewport_cull_2d,
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
    from cellier.v2._state import AxisAlignedSelectionState, DimsState
    from cellier.v2.events._events import (
        AppearanceChangedEvent,
        DataStoreContentsChangedEvent,
        DataStoreMetadataChangedEvent,
        VisualVisibilityChangedEvent,
    )
    from cellier.v2.render.block_cache._tile_manager_2d import (
        BlockKey2D,
    )
    from cellier.v2.render.block_cache._tile_manager_2d import (
        TileSlot as TileSlot2D,
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
    level_shapes : list[tuple[int, ...]]
        Displayed-axis shape at each scale level, finest first.
        For 3D rendering this is ``(D, H, W)``; for 2D it is ``(H, W)``.
        The caller extracts the displayed dimensions before passing.
    downscale_factors : list[int]
        Downscale factor for each level relative to the original data,
        e.g. ``[1, 2, 4]``.  Must have the same length as
        ``level_shapes``.
    block_size : int
        Rendering brick side length in voxels.
    """

    def __init__(
        self,
        level_shapes: list[tuple[int, ...]],
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
        level_shapes: list[tuple[int, ...]],
        block_size: int,
    ) -> VolumeGeometry:
        """Build a ``VolumeGeometry`` from a ``MultiscaleImageVisual`` model.

        Parameters
        ----------
        model : MultiscaleImageVisual
            The visual model.  Only ``downscale_factors`` is read.
        level_shapes : list[tuple[int, ...]]
            Displayed-axis shape per level, finest first.
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

    def _rebuild(self, level_shapes: list[tuple[int, ...]]) -> None:
        self.level_shapes = list(level_shapes)
        self.layouts = [
            BlockLayout3D(volume_shape=shape, block_size=self.block_size)
            for shape in level_shapes
        ]
        self.base_layout = self.layouts[0]
        self._level_grids = build_level_grids(self.base_layout, self.n_levels)

    def update(self, level_shapes: list[tuple[int, ...]]) -> None:
        """Rebuild from new level shapes after a DataStoreMutated event."""
        self._rebuild(level_shapes)


# ---------------------------------------------------------------------------
# ImageGeometry2D
# ---------------------------------------------------------------------------


class ImageGeometry2D:
    """Pre-built metadata cache for a multiscale 2-D image.

    Analogous to ``VolumeGeometry`` but for 2D tile grids.

    Parameters
    ----------
    level_shapes : list[tuple[int, int]]
        Image shape ``(H, W)`` at each scale level, finest first.
        The caller extracts the two displayed dimensions before passing.
    block_size : int
        Tile side length in pixels.
    n_levels : int
        Number of LOD levels.
    """

    def __init__(
        self,
        level_shapes: list[tuple[int, int]],
        block_size: int,
        n_levels: int,
    ) -> None:
        self.block_size = block_size
        self.n_levels = n_levels
        self.level_shapes = list(level_shapes)

        # Build 2D base layout from finest level (H, W).
        self.base_layout = BlockLayout2D.from_shape(
            shape=(level_shapes[0][0], level_shapes[0][1]),
            block_size=block_size,
        )
        self._level_grids = build_tile_grids_2d(self.base_layout, n_levels)

    def update(self, level_shapes: list[tuple[int, int]]) -> None:
        """Rebuild from new level shapes after displayed axes change."""
        self.level_shapes = list(level_shapes)
        self.base_layout = BlockLayout2D.from_shape(
            shape=(level_shapes[0][0], level_shapes[0][1]),
            block_size=self.block_size,
        )
        self._level_grids = build_tile_grids_2d(self.base_layout, self.n_levels)


# ---------------------------------------------------------------------------
# Coordinate helpers
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


def _build_axis_selections(
    sel: AxisAlignedSelectionState,
    ndim: int,
    display_coords: list[tuple[int, int]],
    scale: int,
) -> tuple[int | tuple[int, int], ...]:
    """Map brick/tile window coords and slice indices onto the full nD axis list.

    ``display_coords`` are in the order of ``sel.displayed_axes``.
    Non-displayed axes get their slice index divided by ``scale``
    (coarser levels correspond to smaller indices).
    """
    display_pos = {ax: i for i, ax in enumerate(sel.displayed_axes)}
    result: list[int | tuple[int, int]] = []
    for data_axis in range(ndim):
        if data_axis in display_pos:
            result.append(display_coords[display_pos[data_axis]])
        else:
            result.append(sel.slice_indices[data_axis] // scale)
    return tuple(result)


def _block_key_2d_to_padded_coords(
    key: BlockKey2D,
    block_size: int,
    overlap: int,
) -> tuple[int, int, int, int]:
    """Translate a BlockKey2D to padded pixel coordinates.

    Returns y0, x0, y1, x1.  May extend outside store bounds.
    """
    padded = block_size + 2 * overlap
    y0 = key.gy * block_size - overlap
    x0 = key.gx * block_size - overlap
    return y0, x0, y0 + padded, x0 + padded


# ---------------------------------------------------------------------------
# GFXMultiscaleImageVisual
# ---------------------------------------------------------------------------


class GFXMultiscaleImageVisual:
    """Render-layer wrapper for one logical multiscale image visual.

    Owns GPU resources (brick caches, LUT textures, pygfx nodes) for both
    2D and 3D rendering.  Implements ``build_slice_request()`` /
    ``build_slice_request_2d()`` and ``on_data_ready()`` /
    ``on_data_ready_2d()`` for use by a ``SliceCoordinator``.

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
        volume_geometry: VolumeGeometry | None,
        image_geometry_2d: ImageGeometry2D | None,
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
        self._image_geometry_2d = image_geometry_2d
        self._gpu_budget_bytes = gpu_budget_bytes
        self._frame_number = 0
        self._pending_slot_map: dict[UUID, tuple[BlockKey3D, TileSlot]] = {}
        self._pending_slot_map_2d: dict[UUID, tuple[BlockKey2D, TileSlot2D]] = {}
        self._last_plan_stats: dict = {}

        # ── 3D GPU resources (only when volume_geometry is provided) ───
        self._block_cache_3d: BlockCache3D | None = None
        self._lut_manager_3d: LutIndirectionManager3D | None = None
        self._lut_params_buffer_3d = None
        self._block_scales_buffer_3d = None
        if volume_geometry is not None:
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

        # ── 2D GPU resources (only when image_geometry_2d is provided) ─
        self._block_cache_2d: BlockCache2D | None = None
        self._lut_manager_2d: LutIndirectionManager2D | None = None
        self._lut_params_buffer_2d = None
        self._block_scales_buffer_2d = None
        if image_geometry_2d is not None:
            cache_parameters_2d = compute_block_cache_parameters_2d(
                gpu_budget_bytes=gpu_budget_bytes,
                block_size=image_geometry_2d.block_size,
            )
            self._block_cache_2d = BlockCache2D(cache_parameters=cache_parameters_2d)
            self._lut_manager_2d = LutIndirectionManager2D(
                base_layout=image_geometry_2d.base_layout,
                n_levels=image_geometry_2d.n_levels,
            )
            self._lut_params_buffer_2d = build_lut_params_buffer_2d(
                image_geometry_2d.base_layout, cache_parameters_2d
            )
            self._block_scales_buffer_2d = build_block_scales_buffer_2d(
                image_geometry_2d.n_levels
            )

        if colormap is None:
            colormap = gfx.cm.viridis

        # ── 3D node ─────────────────────────────────────────────────────
        self.node_3d: gfx.Volume | None = None
        self.material_3d: VolumeBlockMaterial | None = None
        self._proxy_tex_3d: gfx.Texture | None = None
        if "3d" in render_modes and volume_geometry is not None:
            self.node_3d, self.material_3d, self._proxy_tex_3d = self._build_3d_node(
                colormap=colormap,
                clim=clim,
                threshold=threshold,
                interpolation=interpolation,
            )

        # ── 2D node ─────────────────────────────────────────────────────
        self.node_2d: gfx.Image | None = None
        self.material_2d: ImageBlockMaterial | None = None
        self._proxy_tex_2d: gfx.Texture | None = None
        if "2d" in render_modes and image_geometry_2d is not None:
            self.node_2d, self.material_2d, self._proxy_tex_2d = self._build_2d_node(
                colormap=colormap, clim=clim, interpolation=interpolation
            )

    @classmethod
    def from_cellier_model(
        cls,
        model: MultiscaleImageVisual,
        level_shapes: list[tuple[int, ...]],
        displayed_axes: tuple[int, ...],
        render_modes: set[str],
        block_size: int = 32,
        gpu_budget_bytes: int = 1 * 1024**3,
        threshold: float = 0.5,
        interpolation: str = "linear",
    ) -> GFXMultiscaleImageVisual:
        """Build a ``GFXMultiscaleImageVisual`` from a ``MultiscaleImageVisual`` model.

        Parameters
        ----------
        model : MultiscaleImageVisual
            Source visual model.
        level_shapes : list[tuple[int, ...]]
            Full nD shape per level, finest first.
        displayed_axes : tuple[int, ...]
            Which axes are displayed (from dims.selection.displayed_axes).
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
        # Extract the displayed-axis shapes.
        displayed_level_shapes = [
            tuple(shape[ax] for ax in displayed_axes) for shape in level_shapes
        ]

        # Build 3D geometry only when 3D rendering is requested.
        volume_geometry: VolumeGeometry | None = None
        if "3d" in render_modes and len(displayed_axes) == 3:
            volume_geometry = VolumeGeometry.from_cellier_model(
                model, displayed_level_shapes, block_size
            )

        # Build 2D geometry only when 2D rendering is requested.
        image_geometry_2d: ImageGeometry2D | None = None
        if "2d" in render_modes and len(displayed_axes) == 2:
            level_shapes_2d = [(s[0], s[1]) for s in displayed_level_shapes]
            image_geometry_2d = ImageGeometry2D(
                level_shapes=level_shapes_2d,
                block_size=block_size,
                n_levels=len(level_shapes),
            )

        colormap = model.appearance.color_map.to_pygfx(N=256)
        clim = model.appearance.clim

        return cls(
            visual_model_id=model.id,
            volume_geometry=volume_geometry,
            image_geometry_2d=image_geometry_2d,
            render_modes=render_modes,
            colormap=colormap,
            clim=clim,
            threshold=threshold,
            interpolation=interpolation,
            gpu_budget_bytes=gpu_budget_bytes,
        )

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def n_levels(self) -> int:
        """Number of LOD levels."""
        if self._volume_geometry is not None:
            return self._volume_geometry.n_levels
        if self._image_geometry_2d is not None:
            return self._image_geometry_2d.n_levels
        raise RuntimeError("No geometry available")

    # ── Geometry rebuild ─────────────────────────────────────────────

    def rebuild_geometry(
        self,
        level_shapes: list[tuple[int, ...]],
        displayed_axes: tuple[int, ...],
    ) -> tuple[gfx.WorldObject | None, gfx.WorldObject | None]:
        """Rebuild geometry and GPU resources after a displayed_axes change.

        Returns ``(old_node, new_node)`` for the active rendering mode so
        the caller can swap the node in the scene graph.

        Parameters
        ----------
        level_shapes : list[tuple[int, ...]]
            Full nD shape per level from the data store.
        displayed_axes : tuple[int, ...]
            The new set of displayed axes.

        Returns
        -------
        tuple[old_node, new_node]
            The previous and replacement pygfx nodes (may be ``None``).
        """
        displayed_level_shapes = [
            tuple(shape[ax] for ax in displayed_axes) for shape in level_shapes
        ]

        old_node: gfx.WorldObject | None = None
        new_node: gfx.WorldObject | None = None

        if "3d" in self.render_modes and len(displayed_axes) == 3:
            old_node = self.node_3d
            if self._volume_geometry is not None:
                self._volume_geometry.update(displayed_level_shapes)
                self._rebuild_3d_resources()
                new_node = self.node_3d

        if "2d" in self.render_modes and len(displayed_axes) == 2:
            old_node = self.node_2d
            shapes_2d = [(s[0], s[1]) for s in displayed_level_shapes]
            if self._image_geometry_2d is not None:
                self._image_geometry_2d.update(shapes_2d)
                self._rebuild_2d_resources()
                new_node = self.node_2d

        return old_node, new_node

    def _rebuild_3d_resources(self) -> None:
        """Rebuild 3D GPU resources after geometry update."""
        geo = self._volume_geometry
        # Clear cache
        self._block_cache_3d.tile_manager.release_all_in_flight()
        # Rebuild LUT manager
        self._lut_manager_3d = LutIndirectionManager3D(
            base_layout=geo.base_layout,
            n_levels=geo.n_levels,
        )
        # Rebuild param buffers
        self._lut_params_buffer_3d = build_lut_params_buffer_3d(
            geo.base_layout,
            self._block_cache_3d.info,
            proxy_voxels_per_brick=geo.block_size,
        )
        self._block_scales_buffer_3d = build_block_scales_buffer_3d(
            geo.downscale_factors
        )
        # Rebuild node preserving current appearance
        if self.node_3d is not None:
            colormap = self.material_3d.map
            clim = self.material_3d.clim
            threshold = self.material_3d.threshold
            interpolation = self.material_3d.interpolation
            self.node_3d, self.material_3d, self._proxy_tex_3d = self._build_3d_node(
                colormap=colormap,
                clim=clim,
                threshold=threshold,
                interpolation=interpolation,
            )
        self._pending_slot_map = {}

    def _rebuild_2d_resources(self) -> None:
        """Rebuild 2D GPU resources after geometry update."""
        geo2d = self._image_geometry_2d
        # Clear cache
        self._block_cache_2d.tile_manager.release_all_in_flight()
        # Rebuild LUT manager
        self._lut_manager_2d = LutIndirectionManager2D(
            base_layout=geo2d.base_layout,
            n_levels=geo2d.n_levels,
        )
        # Rebuild param buffers
        self._lut_params_buffer_2d = build_lut_params_buffer_2d(
            geo2d.base_layout, self._block_cache_2d.info
        )
        self._block_scales_buffer_2d = build_block_scales_buffer_2d(geo2d.n_levels)
        # Rebuild node preserving current appearance
        if self.node_2d is not None:
            colormap = self.material_2d.map
            clim = self.material_2d.clim
            interpolation = self.material_2d.interpolation
            self.node_2d, self.material_2d, self._proxy_tex_2d = self._build_2d_node(
                colormap=colormap, clim=clim, interpolation=interpolation
            )
        self._pending_slot_map_2d = {}

    # ── 3D SliceCoordinator interface ──────────────────────────────────

    def build_slice_request(
        self,
        camera_pos: np.ndarray,
        frustum_planes: np.ndarray | None,
        thresholds: list[float] | None,
        dims_state: DimsState | None = None,
        force_level: int | None = None,
    ) -> list[ChunkRequest]:
        """Run the synchronous 3D planning phase and return ChunkRequests.

        Pipeline: LOAD select -> distance sort -> optional frustum cull ->
        budget cap -> stage() -> build ``ChunkRequest`` objects.
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
            scale = 2 ** (brick_key.level - 1)
            display_coords = [(z0, z1), (y0, y1), (x0, x1)]
            if dims_state is not None:
                ndim = len(dims_state.axis_labels)
                axis_selections = _build_axis_selections(
                    dims_state.selection, ndim, display_coords, scale
                )
            else:
                axis_selections = tuple(display_coords)
            req = ChunkRequest(
                chunk_request_id=chunk_id,
                slice_request_id=slice_id,
                scale_index=brick_key.level - 1,
                axis_selections=axis_selections,
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

        self._last_plan_stats = stats = {
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

        if n_dropped > 0:
            _PERF_LOGGER.warning(
                "budget_exceeded  required=%d  budget=%d  dropped=%d  "
                "(consider larger cache or tighter LOD thresholds)",
                n_needed,
                n_budget,
                n_dropped,
            )

        _PERF_LOGGER.info(
            "[frame %d]  lod_select=%.1fms  dist_sort=%.1fms  frustum_cull=%.1fms  "
            "stage=%.1fms  |  required=%d  culled=%d  hits=%d  misses=%d",
            self._frame_number,
            stats["lod_select_ms"],
            stats["distance_sort_ms"],
            stats.get("frustum_cull_ms", 0.0),
            stats.get("stage_ms", 0.0),
            stats["total_required"],
            stats.get("n_culled", 0),
            stats["hits"],
            stats["misses"],
        )

        return chunk_requests

    def on_data_ready(
        self,
        batch: list[tuple[ChunkRequest, np.ndarray]],
    ) -> None:
        """Commit an arriving batch of 3D bricks to the GPU cache."""
        for req, data in batch:
            entry = self._pending_slot_map.get(req.chunk_request_id)
            if entry is None:
                continue
            brick_key, slot = entry
            self._block_cache_3d.write_brick(slot, data, key=brick_key)
            self._block_cache_3d.tile_manager.commit(brick_key, slot)

        _GPU_LOGGER.info(
            "gpu_flush  bricks_in_batch=%d  resident=%d",
            len(batch),
            self._block_cache_3d.n_resident,
        )

        self._lut_manager_3d.rebuild(self._block_cache_3d.tile_manager)

        _GPU_LOGGER.info(
            "lut_rebuilt  resident=%d  frame=%d",
            self._block_cache_3d.n_resident,
            self._frame_number,
        )

    def cancel_pending(self) -> None:
        """Release all in-flight 3D slots."""
        self._block_cache_3d.tile_manager.release_all_in_flight()
        self._pending_slot_map = {}

    # ── 2D SliceCoordinator interface ──────────────────────────────────

    def build_slice_request_2d(
        self,
        camera_pos: np.ndarray,
        viewport_width_px: float,
        world_width: float,
        view_min: np.ndarray | None,
        view_max: np.ndarray | None,
        dims_state: DimsState,
        lod_bias: float = 1.0,
        force_level: int | None = None,
        use_culling: bool = True,
    ) -> list[ChunkRequest]:
        """Run the synchronous 2D planning phase and return ChunkRequests.

        Pipeline: LOD select -> distance sort -> optional viewport cull ->
        budget cap -> stage() -> build ``ChunkRequest`` objects.

        Parameters
        ----------
        camera_pos : ndarray, shape (3,)
            Camera world-space position ``(x, y, z)``.
        viewport_width_px : float
            Viewport width in logical pixels.
        world_width : float
            Visible world width in world units.
        view_min : ndarray, shape (2,) or None
            Viewport AABB minimum ``(x, y)``.  ``None`` disables culling.
        view_max : ndarray, shape (2,) or None
            Viewport AABB maximum ``(x, y)``.  ``None`` disables culling.
        dims_state : DimsState
            Current dimension display state.
        lod_bias : float
            Multiplicative LOD bias.
        force_level : int or None
            Override: all tiles assigned this 1-based level.
        use_culling : bool
            Enable viewport culling.

        Returns
        -------
        list[ChunkRequest]
            Nearest-first.  Empty when all required tiles are cached.
        """
        t_plan_start = time.perf_counter()
        self._frame_number += 1
        geo2d = self._image_geometry_2d
        block_size = geo2d.block_size
        n_levels = geo2d.n_levels

        # 1. LOD selection
        t0 = time.perf_counter()
        tile_arr = select_lod_2d(
            geo2d._level_grids,
            n_levels,
            viewport_width_px=viewport_width_px,
            world_width=world_width,
            lod_bias=lod_bias,
            force_level=force_level,
        )
        lod_select_ms = (time.perf_counter() - t0) * 1000

        # 2. Distance sort
        t0 = time.perf_counter()
        tile_arr = sort_tiles_by_distance_2d(tile_arr, camera_pos, block_size)
        distance_sort_ms = (time.perf_counter() - t0) * 1000

        # Convert to dict
        required = arr_to_block_keys_2d(tile_arr)
        n_total = len(required)

        # 3. Viewport culling
        n_culled = 0
        cull_ms = 0.0
        if use_culling and view_min is not None and view_max is not None:
            t0 = time.perf_counter()
            required, n_culled = viewport_cull_2d(
                required, block_size, view_min, view_max
            )
            cull_ms = (time.perf_counter() - t0) * 1000

        # 4. Budget truncation
        n_needed = len(required)
        n_budget = self._block_cache_2d.info.n_slots - 1
        n_dropped = max(0, n_needed - n_budget)
        if n_dropped:
            keys_to_keep = list(required.keys())[:n_budget]
            required = {k: required[k] for k in keys_to_keep}

        # 5. Stage
        t0 = time.perf_counter()
        fill_plan = self._block_cache_2d.tile_manager.stage(
            required, self._frame_number
        )
        stage_ms = (time.perf_counter() - t0) * 1000

        # 6. Build ChunkRequests
        slice_id = uuid4()
        chunk_requests: list[ChunkRequest] = []
        self._pending_slot_map_2d = {}

        overlap = self._block_cache_2d.info.overlap

        ndim = len(dims_state.axis_labels)
        sel = dims_state.selection

        for tile_key, slot in fill_plan:
            chunk_id = uuid4()
            y0, x0, y1, x1 = _block_key_2d_to_padded_coords(
                tile_key, block_size, overlap
            )
            scale = 2 ** (tile_key.level - 1)
            display_coords = [(y0, y1), (x0, x1)]
            axis_selections = _build_axis_selections(sel, ndim, display_coords, scale)

            req = ChunkRequest(
                chunk_request_id=chunk_id,
                slice_request_id=slice_id,
                scale_index=tile_key.level - 1,
                axis_selections=axis_selections,
            )
            chunk_requests.append(req)
            self._pending_slot_map_2d[chunk_id] = (tile_key, slot)

        plan_total_ms = (time.perf_counter() - t_plan_start) * 1000

        self._last_plan_stats = stats = {
            "hits": len(required) - len(fill_plan),
            "misses": len(fill_plan),
            "fills": len(fill_plan),
            "total_required": n_total,
            "n_culled": n_culled,
            "n_needed": n_needed,
            "n_budget": n_budget,
            "n_dropped": n_dropped,
            "lod_select_ms": lod_select_ms,
            "distance_sort_ms": distance_sort_ms,
            "cull_ms": cull_ms,
            "stage_ms": stage_ms,
            "plan_total_ms": plan_total_ms,
        }

        if n_dropped > 0:
            _PERF_LOGGER.warning(
                "budget_exceeded  required=%d  budget=%d  dropped=%d  "
                "(consider larger cache or tighter LOD thresholds)",
                n_needed,
                n_budget,
                n_dropped,
            )

        _PERF_LOGGER.info(
            "[frame %d]  lod_select=%.1fms  dist_sort=%.1fms  cull=%.1fms  "
            "stage=%.1fms  |  required=%d  culled=%d  hits=%d  misses=%d",
            self._frame_number,
            stats["lod_select_ms"],
            stats["distance_sort_ms"],
            stats.get("cull_ms", 0.0),
            stats.get("stage_ms", 0.0),
            stats["total_required"],
            stats.get("n_culled", 0),
            stats["hits"],
            stats["misses"],
        )

        return chunk_requests

    def on_data_ready_2d(
        self,
        batch: list[tuple[ChunkRequest, np.ndarray]],
    ) -> None:
        """Commit an arriving batch of 2D tiles to the GPU cache.

        Parameters
        ----------
        batch : list of (ChunkRequest, ndarray)
            Each item is a ``(request, data)`` pair where ``data`` has
            shape ``(pbs, pbs)``, dtype float32.
        """
        for req, data in batch:
            entry = self._pending_slot_map_2d.get(req.chunk_request_id)
            if entry is None:
                continue
            tile_key, slot = entry
            self._block_cache_2d.write_tile(slot, data, key=tile_key)
            self._block_cache_2d.tile_manager.commit(tile_key, slot)

        n_resident = len(self._block_cache_2d.tile_manager.tilemap)

        _GPU_LOGGER.info(
            "gpu_flush  tiles_in_batch=%d  resident=%d",
            len(batch),
            n_resident,
        )

        self._lut_manager_2d.rebuild(self._block_cache_2d.tile_manager)

        _GPU_LOGGER.info(
            "lut_rebuilt  resident=%d  frame=%d",
            n_resident,
            self._frame_number,
        )

    def cancel_pending_2d(self) -> None:
        """Release all in-flight 2D slots."""
        self._block_cache_2d.tile_manager.release_all_in_flight()
        self._pending_slot_map_2d = {}

    # ── EventBus handler methods ─────────────────────────────────────────

    def on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        """Apply GPU-only appearance changes."""
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
        gh, gw = self._image_geometry_2d.base_layout.grid_dims

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

        geometry = gfx.Geometry(grid=proxy_tex)
        image = gfx.Image(geometry, material)

        bs = self._image_geometry_2d.block_size
        image.local.scale = (bs, bs, 1.0)
        image.local.position = (bs * 0.5, bs * 0.5, 0)

        return image, material, proxy_tex
