"""GFXMultiscaleLabelVisual — render-layer visual for multiscale label images."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx

from cellier.v2.data.image import ChunkRequest
from cellier.v2.logging import _GPU_LOGGER, _PERF_LOGGER
from cellier.v2.render._frustum import (
    bricks_in_frustum_arr,
    frustum_planes_from_corners,
)
from cellier.v2.render._level_of_detail import (
    arr_to_brick_keys,
    select_levels_arr_forced,
    select_levels_from_cache,
    sort_arr_by_distance,
)
from cellier.v2.render._level_of_detail_2d import (
    arr_to_block_keys_2d,
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
from cellier.v2.render.lut_indirection import LutIndirectionManager3D
from cellier.v2.render.lut_indirection._lut_buffers_2d import (
    build_block_scales_buffer_2d,
    build_lut_params_buffer_2d,
)
from cellier.v2.render.lut_indirection._lut_indirection_manager_2d import (
    LutIndirectionManager2D,
)
from cellier.v2.render.shaders._label_colormap import (
    build_direct_lut_textures,
    build_label_params_buffer,
)
from cellier.v2.render.shaders._label_multiscale import (
    LabelBlockMaterial,
    LabelVolumeBrickMaterial,
)
from cellier.v2.render.shaders._multiscale_volume_brick import (
    build_brick_scales_buffer,
    build_vol_params_buffer,
    compose_world_transform,
)
from cellier.v2.render.visuals._image import (
    ImageGeometry3D,
    MultiscaleBrickLayout3D,
    _block_key_2d_to_padded_coords,
    _brick_key_to_padded_coords,
    _build_axis_selections,
    _check_transform_no_rotation,
    _norm_size_from_transform,
)
from cellier.v2.render.visuals._image_memory import (
    _box_wireframe_positions,
    _make_aabb_line,
    _pygfx_matrix,
    _rect_wireframe_positions,
)
from cellier.v2.transform import AffineTransform
from cellier.v2.transform._axis_order import select_axes, swap_axes

if TYPE_CHECKING:
    from pygfx.resources import Buffer

    from cellier.v2._state import DimsState
    from cellier.v2.events._events import (
        AABBChangedEvent,
        AppearanceChangedEvent,
        DataStoreContentsChangedEvent,
        DataStoreMetadataChangedEvent,
        TransformChangedEvent,
        VisualVisibilityChangedEvent,
    )
    from cellier.v2.render.block_cache._tile_manager_2d import (
        BlockKey2D,
    )
    from cellier.v2.render.block_cache._tile_manager_2d import (
        TileSlot as TileSlot2D,
    )
    from cellier.v2.visuals._labels import MultiscaleLabelVisual

# Importing this module registers the shader classes with pygfx.
import cellier.v2.render.shaders._label_multiscale as _label_reg  # noqa: F401


class GFXMultiscaleLabelVisual:
    """Render-layer wrapper for one logical multiscale label visual.

    Owns GPU resources (int32 brick caches, LUT textures, label colormap
    buffers, pygfx nodes) for both 2D and 3D rendering.

    Parameters
    ----------
    visual_model_id : UUID
        ID of the associated ``MultiscaleLabelVisual`` model.
    volume_geometry : MultiscaleBrickLayout3D or None
        Pre-built metadata cache (level shapes, LOAD grids). ``None`` skips 3D.
    image_geometry_2d : ImageGeometry3D or None
        Pre-built 2D tile geometry. ``None`` skips 2D.
    render_modes : set[str]
        Which nodes to build: ``{"3d"}``, ``{"2d"}``, or ``{"2d", "3d"}``.
    background_label : int
        Label ID treated as transparent.
    colormap_mode : "random" | "direct"
        Colormap mode (baked into shader at construction).
    salt : int
        Hash seed for random mode.
    color_dict : dict[int, tuple[float, float, float, float]]
        Label ID → RGBA for direct mode.
    render_mode : "iso_categorical" | "flat_categorical"
        3D render mode.
    gpu_budget_bytes_3d : int
        Maximum GPU memory for the 3D brick cache.
    gpu_budget_bytes_2d : int
        Maximum GPU memory for the 2D tile cache.
    """

    cancellable: bool = True

    def __init__(
        self,
        visual_model_id: UUID,
        volume_geometry: MultiscaleBrickLayout3D | None,
        image_geometry_2d: ImageGeometry3D | None,
        render_modes: set[str],
        displayed_axes: tuple[int, ...] | None = None,
        background_label: int = 0,
        colormap_mode: str = "random",
        salt: int = 0,
        color_dict: dict | None = None,
        render_mode: str = "iso_categorical",
        gpu_budget_bytes_3d: int = 1 * 1024**3,
        gpu_budget_bytes_2d: int = 64 * 1024**2,
        transform: AffineTransform | None = None,
        full_level_transforms: list[AffineTransform] | None = None,
        full_level_shapes: list[tuple[int, ...]] | None = None,
        aabb_enabled: bool = False,
        aabb_color: str = "#ffffff",
        aabb_line_width: float = 2.0,
        render_order: int = 0,
        pick_write: bool = True,
    ) -> None:
        self.visual_model_id = visual_model_id

        if full_level_shapes is not None:
            self._ndim = len(full_level_shapes[0])
        elif volume_geometry is not None:
            self._ndim = len(volume_geometry.level_shapes[0])
        elif image_geometry_2d is not None:
            self._ndim = len(image_geometry_2d.level_shapes[0])
        else:
            self._ndim = 3

        if transform is None:
            transform = AffineTransform.identity(ndim=self._ndim)
        elif transform.ndim < self._ndim:
            transform = transform.expand_dims(self._ndim)
        self._transform: AffineTransform = transform
        self.render_modes = render_modes
        self._volume_geometry = volume_geometry
        self._image_geometry_2d = image_geometry_2d

        if full_level_transforms is not None:
            self._level_transforms = list(full_level_transforms)
        elif volume_geometry is not None:
            self._level_transforms = volume_geometry.level_transforms
        elif image_geometry_2d is not None:
            self._level_transforms = image_geometry_2d.level_transforms
        else:
            self._level_transforms = []

        if full_level_shapes is not None:
            self._full_level_shapes = list(full_level_shapes)
        elif volume_geometry is not None:
            self._full_level_shapes = list(volume_geometry.level_shapes)
        elif image_geometry_2d is not None:
            self._full_level_shapes = list(image_geometry_2d.level_shapes)
        else:
            self._full_level_shapes = []

        self._world_to_level_transforms = self._build_world_to_level_transforms()
        self._last_displayed_axes: tuple[int, ...] | None = displayed_axes
        self._gpu_budget_bytes = gpu_budget_bytes_3d
        self._frame_number = 0
        self._pending_slot_map: dict[UUID, tuple[BlockKey3D, TileSlot]] = {}
        self._pending_slot_map_2d: dict[UUID, tuple[BlockKey2D, TileSlot2D]] = {}
        self._last_plan_stats: dict = {}

        self._data_ready_3d: bool = False
        self._data_ready_2d: bool = False

        self._aabb_enabled: bool = aabb_enabled
        self._aabb_color: str = aabb_color
        self._aabb_line_width: float = aabb_line_width

        # Label-specific state
        self._background_label: int = int(background_label)
        self._colormap_mode: str = colormap_mode
        self._salt: int = int(salt)
        self._render_mode: str = render_mode

        # Build label colormap GPU resources
        if color_dict is None:
            color_dict = {}
        keys_tex, colors_tex, n_entries = build_direct_lut_textures(color_dict)
        self._label_keys_texture: gfx.Texture = keys_tex
        self._label_colors_texture: gfx.Texture = colors_tex
        self._n_entries: int = n_entries
        self._label_params_buffer: gfx.Buffer = build_label_params_buffer(
            background_label=self._background_label,
            salt=self._salt,
            n_entries=self._n_entries,
        )

        # ── 3D GPU resources ──────────────────────────────────────────────
        self._block_cache_3d: BlockCache3D | None = None
        self._lut_manager_3d: LutIndirectionManager3D | None = None
        if volume_geometry is not None:
            cache_parameters_3d = compute_block_cache_parameters_3d(
                block_size=volume_geometry.block_size,
                gpu_budget_bytes=gpu_budget_bytes_3d,
                overlap=1,  # BORDER=1: screen-space normals remove gradient probe need
                dtype=np.int32,
            )
            self._block_cache_3d = BlockCache3D(
                cache_parameters=cache_parameters_3d, dtype=np.int32
            )
            self._lut_manager_3d = LutIndirectionManager3D(
                base_layout=volume_geometry.base_layout,
                n_levels=volume_geometry.n_levels,
                level_scale_vecs_data=volume_geometry._scale_vecs_data,
            )

        # ── 2D GPU resources ──────────────────────────────────────────────
        self._block_cache_2d: BlockCache2D | None = None
        self._lut_manager_2d: LutIndirectionManager2D | None = None
        self._lut_params_buffer_2d = None
        self._block_scales_buffer_2d = None
        self._current_slice_coord: tuple[tuple[int, int], ...] | None = None
        if image_geometry_2d is not None:
            cache_parameters_2d = compute_block_cache_parameters_2d(
                gpu_budget_bytes=gpu_budget_bytes_2d,
                block_size=image_geometry_2d.block_size,
            )
            self._block_cache_2d = BlockCache2D(
                cache_parameters=cache_parameters_2d, dtype=np.int32
            )
            self._lut_manager_2d = LutIndirectionManager2D(
                base_layout=image_geometry_2d.base_layout,
                n_levels=image_geometry_2d.n_levels,
                scale_vecs_data=image_geometry_2d._scale_vecs_data,
            )
            self._lut_params_buffer_2d = build_lut_params_buffer_2d(
                image_geometry_2d.base_layout, cache_parameters_2d
            )
            self._block_scales_buffer_2d = build_block_scales_buffer_2d(
                level_scale_vecs_data=image_geometry_2d._scale_vecs_data,
            )

        # ── Brick-shader-specific buffers (3D only) ───────────────────────
        self._vol_params_buffer: Buffer | None = None
        self._brick_scales_buffer: Buffer | None = None
        self._norm_size: np.ndarray | None = None
        self._dataset_size: np.ndarray | None = None
        self._norm_size_axes: tuple[int, ...] | None = None
        if volume_geometry is not None:
            ds = volume_geometry.level_shapes[0]
            self._dataset_size = np.asarray(
                swap_axes(tuple(float(s) for s in ds), (2, 1, 0)),
                dtype=np.float64,
            )
            _check_transform_no_rotation(self._transform)
            axes_for_norm = (
                displayed_axes
                if displayed_axes is not None and len(displayed_axes) == 3
                else tuple(range(self._ndim))[-3:]
            )
            self._norm_size = _norm_size_from_transform(
                self._transform, axes_for_norm, self._dataset_size
            )
            self._norm_size_axes = axes_for_norm
            self._vol_params_buffer = build_vol_params_buffer(
                norm_size=self._norm_size,
                dataset_size=self._dataset_size,
                base_layout=volume_geometry.base_layout,
                cache_info=self._block_cache_3d.info,
            )
            self._brick_scales_buffer = build_brick_scales_buffer(
                volume_geometry._scale_vecs_data
            )

        # ── 3D node ───────────────────────────────────────────────────────
        self.node_3d: gfx.Group | None = None
        self._inner_node_3d: gfx.Volume | None = None
        self.material_3d: LabelVolumeBrickMaterial | None = None
        self._proxy_tex_3d: gfx.Texture | None = None
        self._aabb_line_3d: gfx.Line | None = None
        if "3d" in render_modes and volume_geometry is not None:
            inner, self.material_3d, self._proxy_tex_3d = self._build_3d_node(
                pick_write=pick_write,
            )
            self._inner_node_3d = inner
            self.node_3d = gfx.Group()
            self.node_3d.add(inner)
            self._aabb_line_3d = self._build_aabb_line_3d()
            self.node_3d.add(self._aabb_line_3d)

        # ── 2D node ───────────────────────────────────────────────────────
        self.node_2d: gfx.Group | None = None
        self._inner_node_2d: gfx.Image | None = None
        self.material_2d: LabelBlockMaterial | None = None
        self._proxy_tex_2d: gfx.Texture | None = None
        self._aabb_line_2d: gfx.Line | None = None
        if "2d" in render_modes and image_geometry_2d is not None:
            inner, self.material_2d, self._proxy_tex_2d = self._build_2d_node(
                pick_write=pick_write,
            )
            self._inner_node_2d = inner
            self.node_2d = gfx.Group()
            self.node_2d.add(inner)
            self._aabb_line_2d = self._build_aabb_line_2d()
            self.node_2d.add(self._aabb_line_2d)

        if self.node_3d is not None:
            self.node_3d.render_order = render_order
        if self.node_2d is not None:
            self.node_2d.render_order = render_order

        if self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)

    @classmethod
    def from_cellier_model(
        cls,
        model: MultiscaleLabelVisual,
        level_shapes: list[tuple[int, ...]],
        render_modes: set[str],
        displayed_axes: tuple[int, ...],
    ) -> GFXMultiscaleLabelVisual:
        """Build a ``GFXMultiscaleLabelVisual`` from a model."""
        render_config = model.render_config
        block_size = render_config.block_size
        gpu_budget_bytes = render_config.gpu_budget_bytes
        gpu_budget_bytes_2d = render_config.gpu_budget_bytes_2d

        if len(displayed_axes) == 3:
            axes_3d: tuple[int, ...] | None = displayed_axes
            axes_2d: tuple[int, ...] = displayed_axes[-2:]
        else:
            axes_3d = None
            axes_2d = displayed_axes

        volume_geometry: MultiscaleBrickLayout3D | None = None
        if "3d" in render_modes and axes_3d is not None:
            shapes_3d = [select_axes(s, axes_3d) for s in level_shapes]
            transforms_3d = [t.select_axes(axes_3d) for t in model.level_transforms]
            volume_geometry = MultiscaleBrickLayout3D(
                level_shapes=shapes_3d,
                level_transforms=transforms_3d,
                block_size=block_size,
            )

        image_geometry_2d: ImageGeometry3D | None = None
        if "2d" in render_modes:
            shapes_2d = [select_axes(s, axes_2d) for s in level_shapes]
            transforms_2d = [t.select_axes(axes_2d) for t in model.level_transforms]
            image_geometry_2d = ImageGeometry3D(
                level_shapes=shapes_2d,
                block_size=block_size,
                n_levels=len(level_shapes),
                level_transforms=transforms_2d,
            )

        app = model.appearance
        instance = cls(
            visual_model_id=model.id,
            volume_geometry=volume_geometry,
            image_geometry_2d=image_geometry_2d,
            render_modes=render_modes,
            displayed_axes=displayed_axes,
            background_label=app.background_label,
            colormap_mode=app.colormap_mode,
            salt=app.salt,
            color_dict=dict(app.color_dict),
            render_mode=app.render_mode,
            gpu_budget_bytes_3d=gpu_budget_bytes,
            gpu_budget_bytes_2d=gpu_budget_bytes_2d,
            aabb_enabled=model.aabb.enabled,
            aabb_color=model.aabb.color,
            aabb_line_width=model.aabb.line_width,
            render_order=app.render_order,
            transform=model.transform,
            full_level_transforms=list(model.level_transforms),
            full_level_shapes=list(level_shapes),
        )
        for mat in (instance.material_3d, instance.material_2d):
            if mat is not None:
                mat.opacity = app.opacity
                mat.depth_test = app.depth_test
                mat.depth_write = app.depth_write
                mat.depth_compare = app.depth_compare
                mat.alpha_mode = app.transparency_mode
        return instance

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def n_levels(self) -> int:
        if self._volume_geometry is not None:
            return self._volume_geometry.n_levels
        if self._image_geometry_2d is not None:
            return self._image_geometry_2d.n_levels
        raise RuntimeError("No geometry available")

    # ── Node selection ───────────────────────────────────────────────────

    def get_node_for_dims(self, displayed_axes: tuple[int, ...]) -> gfx.Group | None:
        _old_node, new_node = self.rebuild_geometry(
            self._full_level_shapes, displayed_axes
        )
        return new_node

    # ── Geometry rebuild ─────────────────────────────────────────────────

    def rebuild_geometry(
        self,
        level_shapes: list[tuple[int, ...]],
        displayed_axes: tuple[int, ...],
    ) -> tuple[gfx.WorldObject | None, gfx.WorldObject | None]:
        self._full_level_shapes = list(level_shapes)
        self._last_displayed_axes = displayed_axes

        old_node: gfx.WorldObject | None = None
        new_node: gfx.WorldObject | None = None

        if "3d" in self.render_modes and len(displayed_axes) == 3:
            old_node = self.node_3d
            if self._volume_geometry is not None:
                shapes_3d = [
                    tuple(s[ax] for ax in displayed_axes) for s in level_shapes
                ]
                if shapes_3d != self._volume_geometry.level_shapes:
                    self._volume_geometry.update(shapes_3d)
                    self._rebuild_3d_resources()
                new_node = self.node_3d

        if "2d" in self.render_modes and len(displayed_axes) == 2:
            old_node = self.node_2d
            if self._image_geometry_2d is not None:
                shapes_2d_full = [
                    tuple(s[ax] for ax in displayed_axes) for s in level_shapes
                ]
                shapes_2d = [(s[0], s[1]) for s in shapes_2d_full]
                if shapes_2d != self._image_geometry_2d.level_shapes:
                    self._image_geometry_2d.update(shapes_2d)
                    self._rebuild_2d_resources()
                new_node = self.node_2d

        return old_node, new_node

    def _rebuild_3d_resources(self) -> None:
        geo = self._volume_geometry
        self._block_cache_3d.tile_manager.release_all_in_flight()
        self._lut_manager_3d = LutIndirectionManager3D(
            base_layout=geo.base_layout,
            n_levels=geo.n_levels,
            level_scale_vecs_data=geo._scale_vecs_data,
        )
        if self.node_3d is not None:
            inner, self.material_3d, self._proxy_tex_3d = self._build_3d_node()
            self._inner_node_3d = inner
            self._aabb_line_3d = self._build_aabb_line_3d()
            self.node_3d = gfx.Group()
            self.node_3d.add(inner)
            self.node_3d.add(self._aabb_line_3d)
            if self._last_displayed_axes is not None:
                self._update_node_matrix(self._last_displayed_axes)
        self._pending_slot_map = {}

    def _rebuild_2d_resources(self) -> None:
        geo2d = self._image_geometry_2d
        self._block_cache_2d.tile_manager.release_all_in_flight()
        self._lut_manager_2d = LutIndirectionManager2D(
            base_layout=geo2d.base_layout,
            n_levels=geo2d.n_levels,
            scale_vecs_data=geo2d._scale_vecs_data,
        )
        self._lut_params_buffer_2d = build_lut_params_buffer_2d(
            geo2d.base_layout, self._block_cache_2d.info
        )
        self._block_scales_buffer_2d = build_block_scales_buffer_2d(
            level_scale_vecs_data=geo2d._scale_vecs_data,
        )
        if self.node_2d is not None:
            inner, self.material_2d, self._proxy_tex_2d = self._build_2d_node()
            self._inner_node_2d = inner
            self._aabb_line_2d = self._build_aabb_line_2d()
            self.node_2d = gfx.Group()
            self.node_2d.add(inner)
            self.node_2d.add(self._aabb_line_2d)
            if self._last_displayed_axes is not None:
                self._update_node_matrix(self._last_displayed_axes)
        self._pending_slot_map_2d = {}

    def _update_node_matrix(self, displayed_axes: tuple[int, ...]) -> None:
        self._last_displayed_axes = displayed_axes
        sub = self._transform.select_axes(displayed_axes)

        if self.node_3d is not None:
            data_to_world = _pygfx_matrix(sub)
            m = compose_world_transform(
                data_to_world, self._dataset_size, self._norm_size
            )
            self.node_3d.local.matrix = m

        if self.node_2d is not None:
            self.node_2d.local.matrix = _pygfx_matrix(sub)

    def _build_world_to_level_transforms(self) -> list[AffineTransform]:
        inv_visual = AffineTransform(matrix=self._transform.inverse_matrix)
        result: list[AffineTransform] = []
        for lt in self._level_transforms:
            inv_level = AffineTransform(matrix=lt.inverse_matrix)
            composed = inv_level @ inv_visual
            result.append(composed)
        return result

    # ── 3D planning helpers ───────────────────────────────────────────────

    def build_slice_request(
        self,
        camera_pos_world: np.ndarray,
        frustum_corners_world: np.ndarray | None,
        fov_y_rad: float,
        screen_height_px: float,
        lod_bias: float = 1.0,
        dims_state: DimsState | None = None,
        force_level: int | None = None,
    ) -> list[ChunkRequest]:
        t_plan_start = time.perf_counter()
        self._frame_number += 1
        geo = self._volume_geometry

        if dims_state is not None:
            displayed = dims_state.selection.displayed_axes
            if displayed != self._last_displayed_axes:
                self._update_node_matrix(displayed)

        if self._last_displayed_axes is None:
            raise RuntimeError("build_slice_request requires displayed_axes to be set.")
        if len(self._last_displayed_axes) != 3:
            raise ValueError(
                f"build_slice_request expects 3D display, got "
                f"displayed_axes={self._last_displayed_axes}"
            )
        sub_3d = self._transform.select_axes(self._last_displayed_axes)
        cam_zyx = camera_pos_world[[2, 1, 0]]
        cam_data_zyx = sub_3d.imap_coordinates(cam_zyx.reshape(1, -1)).flatten()
        camera_pos_data = cam_data_zyx[[2, 1, 0]]

        if force_level is None and fov_y_rad > 0:
            focal_half_height_world = (screen_height_px / 2.0) / np.tan(fov_y_rad / 2.0)
            thresholds: list[float] | None = [
                geo._level_scale_factors[k - 1] * focal_half_height_world * lod_bias
                for k in range(1, geo.n_levels)
            ]
        else:
            thresholds = None

        if frustum_corners_world is not None:
            corners_flat = frustum_corners_world.reshape(-1, 3)
            corners_flat_zyx = corners_flat[:, [2, 1, 0]]
            corners_data_flat_zyx = sub_3d.imap_coordinates(corners_flat_zyx)
            corners_data = corners_data_flat_zyx[:, [2, 1, 0]].reshape(
                frustum_corners_world.shape
            )
            frustum_planes = frustum_planes_from_corners(corners_data)
        else:
            frustum_planes = None

        _PERF_LOGGER.debug(
            "[frame %d]  camera_pos_world_xyz=%s  camera_pos_voxel_xyz=%s",
            self._frame_number,
            np.round(camera_pos_world, 1).tolist(),
            np.round(camera_pos_data, 1).tolist(),
        )

        t0 = time.perf_counter()
        if force_level is not None:
            brick_arr = select_levels_arr_forced(
                geo.base_layout, force_level, geo._level_grids
            )
        else:
            brick_arr = select_levels_from_cache(
                geo._level_grids,
                geo.n_levels,
                camera_pos_data,
                thresholds=thresholds,
                base_layout=geo.base_layout,
            )
        lod_select_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        brick_arr = sort_arr_by_distance(
            brick_arr,
            camera_pos_data,
            geo.block_size,
            scale_vecs_shader=geo._scale_arr_shader,
            translation_vecs_shader=geo._translation_arr_shader,
        )
        distance_sort_ms = (time.perf_counter() - t0) * 1000

        n_total = len(brick_arr)

        cull_timings: dict = {}
        n_culled = 0
        frustum_cull_ms = 0.0
        if frustum_planes is not None:
            t0 = time.perf_counter()
            brick_arr, cull_timings = bricks_in_frustum_arr(
                brick_arr,
                geo.block_size,
                frustum_planes,
                level_scale_arr_shader=geo._scale_arr_shader,
                level_translation_arr_shader=geo._translation_arr_shader,
            )
            frustum_cull_ms = (time.perf_counter() - t0) * 1000
            n_culled = n_total - len(brick_arr)

        n_needed = len(brick_arr)
        n_budget = self._block_cache_3d.info.n_slots - 1
        n_dropped = max(0, n_needed - n_budget)
        if n_dropped:
            brick_arr = brick_arr[:n_budget]

        t0 = time.perf_counter()
        sorted_required = arr_to_brick_keys(brick_arr)
        fill_plan = self._block_cache_3d.tile_manager.stage(
            sorted_required, self._frame_number
        )
        stage_ms = (time.perf_counter() - t0) * 1000

        slice_id = uuid4()
        chunk_requests: list[ChunkRequest] = []
        self._pending_slot_map = {}

        for brick_key, slot in fill_plan:
            chunk_id = uuid4()
            z0, y0, x0, z1, y1, x1 = _brick_key_to_padded_coords(
                brick_key, geo.block_size, self._block_cache_3d.info.overlap
            )
            level_index = brick_key.level - 1
            display_coords = [(z0, z1), (y0, y1), (x0, x1)]
            if dims_state is not None:
                ndim = len(dims_state.axis_labels)
                axis_selections = _build_axis_selections(
                    dims_state.selection,
                    ndim,
                    display_coords,
                    level_shape=self._full_level_shapes[level_index],
                    world_to_level_k=self._world_to_level_transforms[level_index],
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

        self._last_plan_stats = stats = {
            "hits": len(sorted_required) - len(fill_plan),
            "misses": len(fill_plan),
            "fills": len(fill_plan),
            "total_required": n_total,
            "n_culled": n_culled,
            "n_needed": n_needed,
            "n_budget": n_budget,
            "n_dropped": n_dropped,
            "cull_timings": cull_timings,
            "lod_select_ms": lod_select_ms,
            "distance_sort_ms": distance_sort_ms,
            "frustum_cull_ms": frustum_cull_ms,
            "stage_ms": stage_ms,
            "plan_total_ms": plan_total_ms,
        }

        if n_dropped > 0:
            _PERF_LOGGER.warning(
                "budget_exceeded  required=%d  budget=%d  dropped=%d",
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
        """Commit an arriving batch of 3D label bricks to the GPU cache."""
        non_bg_bricks = 0
        for req, data in batch:
            entry = self._pending_slot_map.get(req.chunk_request_id)
            if entry is None:
                continue
            brick_key, slot = entry
            # contains_label: 1.0 if any voxel != background, 0.0 if all background.
            contains_label = np.any(data != self._background_label)
            slot.brick_max = 1.0 if contains_label else 0.0
            if contains_label:
                non_bg_bricks += 1
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

        if not self._data_ready_3d and self._aabb_line_3d is not None:
            self._data_ready_3d = True
            self._aabb_line_3d.visible = self._aabb_enabled

    def cancel_pending(self) -> None:
        if self._block_cache_3d is None:
            return
        self._block_cache_3d.tile_manager.release_all_in_flight()
        self._pending_slot_map = {}

    # ── 2D SliceCoordinator interface ─────────────────────────────────────

    def build_slice_request_2d(
        self,
        camera_pos_world: np.ndarray,
        viewport_width_px: float,
        world_width: float,
        view_min_world: np.ndarray | None,
        view_max_world: np.ndarray | None,
        dims_state: DimsState,
        lod_bias: float = 1.0,
        force_level: int | None = None,
        use_culling: bool = True,
    ) -> list[ChunkRequest]:
        t_plan_start = time.perf_counter()
        self._frame_number += 1

        self._current_slice_coord = tuple(
            sorted(dims_state.selection.slice_indices.items())
        )

        geo2d = self._image_geometry_2d
        block_size = geo2d.block_size
        n_levels = geo2d.n_levels

        displayed = dims_state.selection.displayed_axes
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)

        sub_2d = self._transform.select_axes(displayed)

        nd2 = sub_2d.ndim
        world_units_per_voxel_2d = np.abs(np.diag(sub_2d.matrix[:nd2, :nd2]))
        world_to_voxel_scale_2d = float(
            np.prod(1.0 / world_units_per_voxel_2d) ** (1.0 / nd2)
        )
        voxel_width = world_width * world_to_voxel_scale_2d

        camera_pos_2d = sub_2d.imap_coordinates(
            camera_pos_world[[1, 0]].reshape(1, -1)
        ).flatten()[[1, 0]]
        camera_pos = np.array(
            [camera_pos_2d[0], camera_pos_2d[1], 0.0], dtype=np.float32
        )

        if use_culling and view_min_world is not None and view_max_world is not None:
            cx = float(camera_pos_world[0])
            cy = float(camera_pos_world[1])
            half_w = world_width / 2.0
            half_h = (float(view_max_world[1]) - float(view_min_world[1])) / 2.0
            corners_world_2d = np.array(
                [
                    [cy - half_h, cx - half_w],
                    [cy - half_h, cx + half_w],
                    [cy + half_h, cx + half_w],
                    [cy + half_h, cx - half_w],
                ],
                dtype=np.float32,
            )
            corners_data_2d = sub_2d.imap_coordinates(corners_world_2d)[:, [1, 0]]
            view_min = corners_data_2d.min(axis=0)
            view_max = corners_data_2d.max(axis=0)
        else:
            view_min = None
            view_max = None

        t0 = time.perf_counter()
        tile_arr = select_lod_2d(
            geo2d._level_grids,
            n_levels,
            viewport_width_px=viewport_width_px,
            voxel_width=voxel_width,
            lod_bias=lod_bias,
            force_level=force_level,
            level_scale_factors=geo2d._level_scale_factors,
        )
        lod_select_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        tile_arr = sort_tiles_by_distance_2d(
            tile_arr,
            camera_pos,
            block_size,
            level_scale_arr_shader=geo2d._scale_arr_shader,
            level_translation_arr_shader=geo2d._translation_arr_shader,
        )
        distance_sort_ms = (time.perf_counter() - t0) * 1000

        required = arr_to_block_keys_2d(tile_arr, slice_coord=self._current_slice_coord)
        n_total = len(required)

        n_culled = 0
        cull_ms = 0.0
        if use_culling and view_min is not None and view_max is not None:
            t0 = time.perf_counter()
            required, n_culled = viewport_cull_2d(
                required,
                block_size,
                view_min,
                view_max,
                level_scale_arr_shader=geo2d._scale_arr_shader,
                level_translation_arr_shader=geo2d._translation_arr_shader,
            )
            cull_ms = (time.perf_counter() - t0) * 1000

        n_needed = len(required)
        n_budget = self._block_cache_2d.info.n_slots - 1
        n_dropped = max(0, n_needed - n_budget)
        if n_dropped:
            keys_to_keep = list(required.keys())[:n_budget]
            required = {k: required[k] for k in keys_to_keep}

        target_level = int(tile_arr[0, 0]) if len(tile_arr) > 0 else 1
        self._block_cache_2d.tile_manager.evict_finer_than(target_level)

        t0 = time.perf_counter()
        fill_plan = self._block_cache_2d.tile_manager.stage(
            required, self._frame_number
        )
        stage_ms = (time.perf_counter() - t0) * 1000

        if not fill_plan:
            self._lut_manager_2d.rebuild(
                self._block_cache_2d.tile_manager,
                current_slice_coord=self._current_slice_coord,
            )

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
            level_index = tile_key.level - 1
            display_coords = [(y0, y1), (x0, x1)]
            axis_selections = _build_axis_selections(
                sel,
                ndim,
                display_coords,
                level_shape=self._full_level_shapes[level_index],
                world_to_level_k=self._world_to_level_transforms[level_index],
            )
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
                "budget_exceeded  required=%d  budget=%d  dropped=%d",
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
        """Commit an arriving batch of 2D label tiles to the GPU cache."""
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

        self._lut_manager_2d.rebuild(
            self._block_cache_2d.tile_manager,
            current_slice_coord=self._current_slice_coord,
        )

        _GPU_LOGGER.info(
            "lut_rebuilt  resident=%d  frame=%d",
            n_resident,
            self._frame_number,
        )

        if not self._data_ready_2d and self._aabb_line_2d is not None:
            self._data_ready_2d = True
            self._aabb_line_2d.visible = self._aabb_enabled

    def cancel_pending_2d(self) -> None:
        if self._block_cache_2d is None:
            return
        self._block_cache_2d.tile_manager.release_all_in_flight()
        self._pending_slot_map_2d = {}

    def invalidate_2d_cache(self) -> None:
        if self._block_cache_2d is None or self._lut_manager_2d is None:
            return
        self.cancel_pending_2d()

    # ── EventBus handler methods ──────────────────────────────────────────

    def on_transform_changed(self, event: TransformChangedEvent) -> None:
        self._transform = event.transform
        self._world_to_level_transforms = self._build_world_to_level_transforms()

        if self._dataset_size is not None:
            _check_transform_no_rotation(self._transform)
            self._norm_size = _norm_size_from_transform(
                self._transform, self._norm_size_axes, self._dataset_size
            )
            if self._vol_params_buffer is not None:
                buf_data = self._vol_params_buffer.data
                buf_data["norm_size_x"] = float(self._norm_size[0])
                buf_data["norm_size_y"] = float(self._norm_size[1])
                buf_data["norm_size_z"] = float(self._norm_size[2])
                self._vol_params_buffer.update_full()

        if self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)

    def on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        """Apply appearance changes for label-specific and base fields."""
        field = event.field_name
        val = event.new_value

        if field == "opacity":
            for mat in (self.material_3d, self.material_2d):
                if mat is not None:
                    mat.opacity = val
        elif field in ("depth_test", "depth_write", "depth_compare"):
            for mat in (self.material_3d, self.material_2d):
                if mat is not None:
                    setattr(mat, field, val)
        elif field == "transparency_mode":
            for mat in (self.material_3d, self.material_2d):
                if mat is not None:
                    mat.alpha_mode = val
        elif field == "render_order":
            if self.node_3d is not None:
                self.node_3d.render_order = val
            if self.node_2d is not None:
                self.node_2d.render_order = val
        elif field == "background_label":
            self._background_label = int(val)
            self._label_params_buffer.data["background_label"] = np.int32(val)
            self._label_params_buffer.update_full()
        elif field == "salt":
            self._salt = int(val)
            self._label_params_buffer.data["salt"] = np.uint32(int(val) & 0xFFFFFFFF)
            self._label_params_buffer.update_full()
        elif field == "color_dict":
            keys_tex, colors_tex, n_entries = build_direct_lut_textures(dict(val))
            self._label_keys_texture = keys_tex
            self._label_colors_texture = colors_tex
            self._n_entries = n_entries
            self._label_params_buffer.data["n_entries"] = np.uint32(n_entries)
            self._label_params_buffer.update_full()
            # Update material texture references so pygfx picks them up next render.
            if self.material_3d is not None:
                self.material_3d.label_keys_texture = keys_tex
                self.material_3d.label_colors_texture = colors_tex
                self.material_3d.n_entries = n_entries
            if self.material_2d is not None:
                self.material_2d.label_keys_texture = keys_tex
                self.material_2d.label_colors_texture = colors_tex
                self.material_2d.n_entries = n_entries
        elif field in ("colormap_mode", "render_mode"):
            import warnings

            warnings.warn(
                f"LabelAppearance.{field} is frozen after construction; "
                "create a new visual to change it.",
                stacklevel=2,
            )
        # LOD fields — these affect planning, not GPU state; nothing to push.

    def on_visibility_changed(self, event: VisualVisibilityChangedEvent) -> None:
        if self.node_3d is not None:
            self.node_3d.visible = event.visible
        if self.node_2d is not None:
            self.node_2d.visible = event.visible

    def on_aabb_changed(self, event: AABBChangedEvent) -> None:
        if event.field_name == "enabled":
            self._aabb_enabled = event.new_value
            if self._aabb_line_3d is not None:
                self._aabb_line_3d.visible = event.new_value and self._data_ready_3d
            if self._aabb_line_2d is not None:
                self._aabb_line_2d.visible = event.new_value and self._data_ready_2d
        elif event.field_name == "color":
            self._aabb_color = event.new_value
            for line in (self._aabb_line_3d, self._aabb_line_2d):
                if line is not None:
                    line.material.color = event.new_value
        elif event.field_name == "line_width":
            self._aabb_line_width = event.new_value
            for line in (self._aabb_line_3d, self._aabb_line_2d):
                if line is not None:
                    line.material.thickness = event.new_value

    def on_data_store_contents_changed(
        self, event: DataStoreContentsChangedEvent
    ) -> None:
        pass

    def on_data_store_metadata_changed(
        self, event: DataStoreMetadataChangedEvent
    ) -> None:
        pass

    def tick(self) -> None:
        pass

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_aabb_line_3d(self) -> gfx.Line:
        if self._norm_size is not None:
            half = self._norm_size / 2.0
            positions = _box_wireframe_positions(-half, half)
        else:
            positions = _box_wireframe_positions(np.zeros(3), np.ones(3))
        return _make_aabb_line(positions, self._aabb_color, self._aabb_line_width)

    def _build_aabb_line_2d(self) -> gfx.Line:
        if self._image_geometry_2d is not None:
            h, w = self._image_geometry_2d.level_shapes[0]
            positions = _rect_wireframe_positions(
                np.array([0.0, 0.0]),
                np.array([float(w), float(h)]),
            )
        else:
            positions = _rect_wireframe_positions(np.zeros(2), np.ones(2))
        return _make_aabb_line(positions, self._aabb_color, self._aabb_line_width)

    def _build_3d_node(
        self,
        pick_write: bool = True,
    ) -> tuple[gfx.Volume, LabelVolumeBrickMaterial, gfx.Texture]:
        """Construct the proxy texture, label brick material, and Volume node."""
        proxy_data = np.zeros((2, 2, 2), dtype=np.float32)
        proxy_tex = gfx.Texture(proxy_data, dim=3)

        keys_tex = self._label_keys_texture if self._colormap_mode == "direct" else None
        colors_tex = (
            self._label_colors_texture if self._colormap_mode == "direct" else None
        )

        material = LabelVolumeBrickMaterial(
            cache_texture=self._block_cache_3d.cache_tex,
            lut_texture=self._lut_manager_3d.lut_tex,
            brick_max_texture=self._lut_manager_3d.brick_max_tex,
            vol_params_buffer=self._vol_params_buffer,
            block_scales_buffer=self._brick_scales_buffer,
            label_params_buffer=self._label_params_buffer,
            label_keys_texture=keys_tex,
            label_colors_texture=colors_tex,
            background_label=self._background_label,
            colormap_mode=self._colormap_mode,
            salt=self._salt,
            render_mode=self._render_mode,
            n_entries=self._n_entries,
            pick_write=pick_write,
        )

        geometry = gfx.Geometry(grid=proxy_tex)
        vol = gfx.Volume(geometry, material)
        return vol, material, proxy_tex

    def _build_2d_node(
        self,
        pick_write: bool = True,
    ) -> tuple[gfx.Image, LabelBlockMaterial, gfx.Texture]:
        gh, gw = self._image_geometry_2d.base_layout.grid_dims

        proxy_data = np.zeros((gh, gw), dtype=np.float32)
        proxy_tex = gfx.Texture(proxy_data, dim=2)

        keys_tex = self._label_keys_texture if self._colormap_mode == "direct" else None
        colors_tex = (
            self._label_colors_texture if self._colormap_mode == "direct" else None
        )

        material = LabelBlockMaterial(
            cache_texture=self._block_cache_2d.cache_tex,
            lut_texture=self._lut_manager_2d.lut_tex,
            lut_params_buffer=self._lut_params_buffer_2d,
            block_scales_buffer=self._block_scales_buffer_2d,
            label_params_buffer=self._label_params_buffer,
            label_keys_texture=keys_tex,
            label_colors_texture=colors_tex,
            colormap_mode=self._colormap_mode,
            background_label=self._background_label,
            n_entries=self._n_entries,
            pick_write=pick_write,
        )

        geometry = gfx.Geometry(grid=proxy_tex)
        image = gfx.Image(geometry, material)

        bs = self._image_geometry_2d.block_size
        h, w = self._image_geometry_2d.level_shapes[0]
        sx = float(w) / float(gw * bs)
        sy = float(h) / float(gh * bs)
        scale_x = float(bs) * sx
        scale_y = float(bs) * sy
        image.local.scale = (scale_x, scale_y, 1.0)
        image.local.position = (scale_x * 0.5, scale_y * 0.5, 0.0)

        return image, material, proxy_tex
