"""GFXMultiscaleImageVisual — render-layer visual for multiscale images."""

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
from cellier.v2.render.shaders._multiscale_volume_brick import (
    MultiscaleVolumeBrickMaterial,
    build_brick_scales_buffer,
    build_vol_params_buffer,
    compose_world_transform,
    compute_normalized_size,
)
from cellier.v2.render.visuals._image_memory import (
    _box_wireframe_positions,
    _make_aabb_line,
    _pygfx_matrix,
    _rect_wireframe_positions,
)
from cellier.v2.transform import AffineTransform

if TYPE_CHECKING:
    from pygfx.resources import Buffer

    from cellier.v2._state import AxisAlignedSelectionState, DimsState
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
    from cellier.v2.visuals._image import MultiscaleImageVisual

# Importing these modules registers shader classes with pygfx via the
# @register_wgpu_render_function decorator.
import cellier.v2.render.shaders._block_volume as _shader_reg  # noqa: F401
import cellier.v2.render.shaders._multiscale_volume_brick as _brick_reg  # noqa: F401

# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------


def _extract_scale_and_translation(
    level_transforms: list[AffineTransform],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract per-axis scale and translation vectors from level transforms.

    Vectors are returned in data-axis order ``(axis0, axis1, ...) =
    (z, y, x)``, matching the ``AffineTransform`` matrix convention.
    Callers that feed these into world-space geometry functions must
    convert to shader order ``(x, y, z)`` first.

    Parameters
    ----------
    level_transforms : list[AffineTransform]
        Per-level transforms mapping level-k voxel coords to level-0.

    Returns
    -------
    scale_vecs : list[np.ndarray]
        ``(ndim,)`` per level — diagonal of the spatial block.
    translation_vecs : list[np.ndarray]
        ``(ndim,)`` per level — translation column.
    """
    scale_vecs: list[np.ndarray] = []
    translation_vecs: list[np.ndarray] = []
    for t in level_transforms:
        nd = t.ndim
        scale_vecs.append(np.diag(t.matrix[:nd, :nd]).copy())
        translation_vecs.append(t.matrix[:nd, nd].copy())
    return scale_vecs, translation_vecs


# ---------------------------------------------------------------------------
# Brick-shader transform helpers
# ---------------------------------------------------------------------------


def _check_transform_no_rotation(transform: AffineTransform) -> None:
    """Raise ValueError if the transform contains rotation or shear.

    The brick shader only supports scale and translation.  A transform with
    off-diagonal entries in the linear submatrix (i.e. rotation or shear)
    would require ``norm_to_voxel`` to be a full affine inverse, which the
    current WGSL does not implement.

    Parameters
    ----------
    transform : AffineTransform
        The data-to-world transform to validate.

    Raises
    ------
    ValueError
        If the linear part of the transform is not diagonal.
    """
    nd = transform.ndim
    linear = transform.matrix[:nd, :nd]
    diagonal_only = np.diag(np.diag(linear))
    if not np.allclose(linear, diagonal_only, atol=1e-5):
        raise ValueError(
            "The brick shader only supports scale and translation transforms. "
            "The provided transform contains rotation or shear components. "
            "Support for general affine transforms requires changes to the "
            "WGSL norm_to_voxel function and is not yet implemented."
        )


def _norm_size_from_transform(
    transform: AffineTransform,
    dataset_size_xyz: np.ndarray,
) -> np.ndarray:
    """Compute normalised physical size from a scale+translation transform.

    Extracts per-axis physical scale factors from the column norms of the
    linear submatrix (data-axis order: z, y, x for 3-D), converts to shader
    order (x=W, y=H, z=D), multiplies by ``dataset_size_xyz``, and
    normalises so the longest axis equals 1.0.

    For a pure diagonal (scale-only) transform the column norms equal the
    absolute diagonal values, so this is exact.

    Parameters
    ----------
    transform : AffineTransform
        The data-to-world transform.  Must be scale + translation only
        (validated separately by ``_check_transform_no_rotation``).
    dataset_size_xyz : ndarray, shape (3,)
        Finest-level voxel counts in shader order (x=W, y=H, z=D).

    Returns
    -------
    ndarray, shape (3,)
        Normalised physical size in shader order, with the longest axis
        equal to 1.0.
    """
    nd = transform.ndim
    linear = transform.matrix[:nd, :nd]  # data order: (z, y, x) for 3-D
    # Column norms give the physical scale factor for each data axis.
    col_norms = np.array(
        [np.linalg.norm(linear[:, i]) for i in range(nd)], dtype=np.float64
    )
    # col_norms is in data order (sz, sy, sx); shader order is (sx, sy, sz).
    per_axis_scale_xyz = col_norms[::-1]
    return compute_normalized_size(dataset_size_xyz, per_axis_scale_xyz)


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
    level_transforms : list[AffineTransform]
        Per-level transforms mapping level-k voxel coords to level-0.
        ``level_transforms[0]`` must be the identity.
    block_size : int
        Rendering brick side length in voxels.
    """

    def __init__(
        self,
        level_shapes: list[tuple[int, ...]],
        level_transforms: list[AffineTransform],
        block_size: int,
    ) -> None:
        self.level_transforms = list(level_transforms)
        self.block_size = block_size
        self.n_levels = len(level_shapes)

        ndim = level_transforms[0].ndim
        assert np.allclose(
            level_transforms[0].matrix, np.eye(ndim + 1)
        ), "level_transforms[0] must be the identity"

        # Data-axis order (z, y, x).
        sv_data, tv_data = _extract_scale_and_translation(level_transforms)
        self._scale_vecs_data = sv_data
        self._translation_vecs_data = tv_data

        # Shader/pygfx order (x=W, y=H, z=D).
        self._scale_vecs_shader = [sv[[2, 1, 0]] for sv in sv_data]
        self._translation_vecs_shader = [tv[[2, 1, 0]] for tv in tv_data]

        # (n_levels, 3) arrays for vectorised hot-path lookups.
        self._scale_arr_shader = np.stack(self._scale_vecs_shader, axis=0)
        self._translation_arr_shader = np.stack(self._translation_vecs_shader, axis=0)

        self._rebuild(level_shapes)

    def _rebuild(self, level_shapes: list[tuple[int, ...]]) -> None:
        self.level_shapes = list(level_shapes)
        self.layouts = [
            BlockLayout3D(volume_shape=shape, block_size=self.block_size)
            for shape in level_shapes
        ]
        self.base_layout = self.layouts[0]
        self._level_grids = build_level_grids(
            self.base_layout,
            self.n_levels,
            level_shapes=self.level_shapes,
            scale_vecs_shader=self._scale_vecs_shader,
            translation_vecs_shader=self._translation_vecs_shader,
        )

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
    level_transforms : list[AffineTransform]
        Per-level transforms mapping level-k voxel coords to level-0
        in the 2D displayed-axis sub-space.
    """

    def __init__(
        self,
        level_shapes: list[tuple[int, int]],
        block_size: int,
        n_levels: int,
        level_transforms: list[AffineTransform] | None = None,
    ) -> None:
        self.block_size = block_size
        self.n_levels = n_levels
        self.level_shapes = list(level_shapes)

        # Build 2D transforms if not provided (fallback to identity).
        if level_transforms is None:
            level_transforms = [
                AffineTransform.identity(ndim=2) for _ in range(n_levels)
            ]
        self.level_transforms = list(level_transforms)

        # Data-axis order (H, W).
        sv_data, tv_data = _extract_scale_and_translation(self.level_transforms)
        self._scale_vecs_data = sv_data
        self._translation_vecs_data = tv_data

        # Shader order (x=W, y=H) — 2D reversal: sv[[1, 0]].
        self._scale_vecs_shader = [sv[[1, 0]] for sv in sv_data]
        self._translation_vecs_shader = [tv[[1, 0]] for tv in tv_data]

        self._scale_arr_shader = np.stack(self._scale_vecs_shader, axis=0)
        self._translation_arr_shader = np.stack(self._translation_vecs_shader, axis=0)

        # Scalar LOD factor per level (geometric mean of per-axis scales).
        self._level_scale_factors = [float(np.sqrt(np.prod(sv))) for sv in sv_data]

        # Build 2D base layout from finest level (H, W).
        self.base_layout = BlockLayout2D.from_shape(
            shape=(level_shapes[0][0], level_shapes[0][1]),
            block_size=block_size,
        )
        self._level_grids = build_tile_grids_2d(
            self.base_layout,
            n_levels,
            level_shapes=self.level_shapes,
            scale_vecs_shader=self._scale_vecs_shader,
            translation_vecs_shader=self._translation_vecs_shader,
        )

    def update(self, level_shapes: list[tuple[int, int]]) -> None:
        """Rebuild from new level shapes after displayed axes change."""
        self.level_shapes = list(level_shapes)
        self.base_layout = BlockLayout2D.from_shape(
            shape=(level_shapes[0][0], level_shapes[0][1]),
            block_size=self.block_size,
        )
        self._level_grids = build_tile_grids_2d(
            self.base_layout,
            self.n_levels,
            level_shapes=self.level_shapes,
            scale_vecs_shader=self._scale_vecs_shader,
            translation_vecs_shader=self._translation_vecs_shader,
        )


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
    level_shape: tuple[int, ...],
    world_to_level_k: AffineTransform,
) -> tuple[int | tuple[int, int], ...]:
    """Map brick/tile window coords and slice indices onto the full nD axis list.

    ``display_coords`` are in the order of ``sel.displayed_axes``.
    Non-displayed axes use ``world_to_level_k.map_coordinates`` to
    map slice indices from world space to level-k voxel space.

    Parameters
    ----------
    sel : AxisAlignedSelectionState
        Current selection state.
    ndim : int
        Number of data dimensions.
    display_coords : list[tuple[int, int]]
        Padded coordinate ranges for displayed axes.
    level_shape : tuple[int, ...]
        Shape of the data at this level.
    world_to_level_k : AffineTransform
        Composed world→level-k transform (data-axis order).
    """
    display_pos = {ax: i for i, ax in enumerate(sel.displayed_axes)}

    # Build N-D world-space point from slice indices.
    world_pt = np.zeros(ndim, dtype=np.float64)
    for ax, idx in sel.slice_indices.items():
        world_pt[ax] = float(idx)

    # Map to level-k voxel coords in one call.
    level_k_pt = world_to_level_k.map_coordinates(world_pt.reshape(1, -1)).flatten()

    result: list[int | tuple[int, int]] = []
    for data_axis in range(ndim):
        if data_axis in display_pos:
            result.append(display_coords[display_pos[data_axis]])
        else:
            raw = float(level_k_pt[data_axis])
            clamped = int(round(raw))
            clamped = max(0, min(clamped, level_shape[data_axis] - 1))
            result.append(clamped)
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
    gpu_budget_bytes_3d : int
        Maximum GPU memory for the 3D brick cache texture.
    gpu_budget_bytes_2d : int
        Maximum GPU memory for the 2D tile cache texture.
    """

    cancellable: bool = True

    def __init__(
        self,
        visual_model_id: UUID,
        volume_geometry: VolumeGeometry | None,
        image_geometry_2d: ImageGeometry2D | None,
        render_modes: set[str],
        displayed_axes: tuple[int, ...] | None = None,
        colormap: gfx.TextureMap | None = None,
        clim: tuple[float, float] = (0.0, 1.0),
        threshold: float = 0.5,
        interpolation: str = "linear",
        gpu_budget_bytes_3d: int = 1 * 1024**3,
        gpu_budget_bytes_2d: int = 64 * 1024**2,
        transform: AffineTransform | None = None,
        full_level_transforms: list[AffineTransform] | None = None,
        full_level_shapes: list[tuple[int, ...]] | None = None,
        use_brick_shader: bool = False,
        aabb_enabled: bool = False,
        aabb_color: str = "#ffffff",
        aabb_line_width: float = 2.0,
        render_order: int = 0,
    ) -> None:
        self.visual_model_id = visual_model_id

        # ndim of the original data (not the displayed subspace).
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

        # Full-ndim level transforms for composed world→level-k mapping.
        # These are always in the original data dimensionality, separate
        # from the displayed-subspace transforms on the geometry objects.
        if full_level_transforms is not None:
            self._level_transforms = list(full_level_transforms)
        elif volume_geometry is not None:
            self._level_transforms = volume_geometry.level_transforms
        elif image_geometry_2d is not None:
            self._level_transforms = image_geometry_2d.level_transforms
        else:
            self._level_transforms = []

        # Full-ndim level shapes for _build_axis_selections (clamping
        # non-displayed slice indices against the correct dimension).
        if full_level_shapes is not None:
            self._full_level_shapes = list(full_level_shapes)
        elif volume_geometry is not None:
            self._full_level_shapes = list(volume_geometry.level_shapes)
        elif image_geometry_2d is not None:
            self._full_level_shapes = list(image_geometry_2d.level_shapes)
        else:
            self._full_level_shapes = []

        self._world_to_level_transforms = self._build_world_to_level_transforms()
        # Track displayed axes for node matrix updates.
        self._last_displayed_axes: tuple[int, ...] | None = displayed_axes
        self._gpu_budget_bytes = gpu_budget_bytes_3d
        self._frame_number = 0
        self._use_brick_shader = use_brick_shader
        self._pending_slot_map: dict[UUID, tuple[BlockKey3D, TileSlot]] = {}
        self._pending_slot_map_2d: dict[UUID, tuple[BlockKey2D, TileSlot2D]] = {}
        self._last_plan_stats: dict = {}

        # Data-ready flags: AABB visibility is suppressed until the first
        # brick/tile batch arrives.
        self._data_ready_3d: bool = False
        self._data_ready_2d: bool = False

        # Cache AABB params so on_aabb_changed can update them.
        self._aabb_enabled: bool = aabb_enabled
        self._aabb_color: str = aabb_color
        self._aabb_line_width: float = aabb_line_width

        # ── 3D GPU resources (only when volume_geometry is provided) ───
        self._block_cache_3d: BlockCache3D | None = None
        self._lut_manager_3d: LutIndirectionManager3D | None = None
        self._lut_params_buffer_3d = None
        self._block_scales_buffer_3d = None
        if volume_geometry is not None:
            cache_parameters_3d = compute_block_cache_parameters_3d(
                block_size=volume_geometry.block_size,
                gpu_budget_bytes=gpu_budget_bytes_3d,
                overlap=3,
            )
            self._block_cache_3d = BlockCache3D(cache_parameters=cache_parameters_3d)
            self._lut_manager_3d = LutIndirectionManager3D(
                base_layout=volume_geometry.base_layout,
                n_levels=volume_geometry.n_levels,
                level_scale_vecs_data=volume_geometry._scale_vecs_data,
            )
            self._lut_params_buffer_3d = build_lut_params_buffer_3d(
                volume_geometry.base_layout,
                cache_parameters_3d,
                proxy_voxels_per_brick=volume_geometry.block_size,
            )
            self._block_scales_buffer_3d = build_block_scales_buffer_3d(
                volume_geometry._scale_vecs_data
            )

        # ── 2D GPU resources (only when image_geometry_2d is provided) ─
        self._block_cache_2d: BlockCache2D | None = None
        self._lut_manager_2d: LutIndirectionManager2D | None = None
        self._lut_params_buffer_2d = None
        self._block_scales_buffer_2d = None
        # Sorted (axis_index, world_value) pairs for the most-recently-requested slice.
        # Used by the two-phase LUT rebuild to separate current-slice tiles from
        # old-slice fallback tiles.
        self._current_slice_coord: tuple[tuple[int, int], ...] | None = None
        if image_geometry_2d is not None:
            cache_parameters_2d = compute_block_cache_parameters_2d(
                gpu_budget_bytes=gpu_budget_bytes_2d,
                block_size=image_geometry_2d.block_size,
            )
            self._block_cache_2d = BlockCache2D(cache_parameters=cache_parameters_2d)
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

        # ── Brick-shader-specific buffers (3D only) ──────────────────────
        self._vol_params_buffer: Buffer | None = None
        self._brick_scales_buffer: Buffer | None = None
        self._norm_size: np.ndarray | None = None
        self._dataset_size: np.ndarray | None = None
        if use_brick_shader and volume_geometry is not None:
            # dataset_size in shader order (x=W, y=H, z=D).
            ds = volume_geometry.level_shapes[0]  # (D, H, W)
            self._dataset_size = np.array(
                [float(ds[2]), float(ds[1]), float(ds[0])], dtype=np.float64
            )
            _check_transform_no_rotation(self._transform)
            self._norm_size = _norm_size_from_transform(
                self._transform, self._dataset_size
            )
            self._vol_params_buffer = build_vol_params_buffer(
                norm_size=self._norm_size,
                dataset_size=self._dataset_size,
                base_layout=volume_geometry.base_layout,
                cache_info=self._block_cache_3d.info,
            )
            self._brick_scales_buffer = build_brick_scales_buffer(
                volume_geometry._scale_vecs_data
            )

        if colormap is None:
            colormap = gfx.cm.viridis

        # ── 3D node ─────────────────────────────────────────────────────
        self.node_3d: gfx.Group | None = None
        self._inner_node_3d: gfx.Volume | None = None
        self.material_3d: VolumeBlockMaterial | MultiscaleVolumeBrickMaterial | None = (
            None
        )
        self._proxy_tex_3d: gfx.Texture | None = None
        self._aabb_line_3d: gfx.Line | None = None
        if "3d" in render_modes and volume_geometry is not None:
            if use_brick_shader:
                inner, self.material_3d, self._proxy_tex_3d = self._build_3d_node_brick(
                    colormap=colormap,
                    clim=clim,
                    threshold=threshold,
                )
            else:
                inner, self.material_3d, self._proxy_tex_3d = self._build_3d_node(
                    colormap=colormap,
                    clim=clim,
                    threshold=threshold,
                    interpolation=interpolation,
                )
            self._inner_node_3d = inner
            self.node_3d = gfx.Group()
            self.node_3d.add(inner)
            # Build AABB line with known geometry (geometry available at construction).
            self._aabb_line_3d = self._build_aabb_line_3d()
            self.node_3d.add(self._aabb_line_3d)

        # ── 2D node ─────────────────────────────────────────────────────
        self.node_2d: gfx.Group | None = None
        self._inner_node_2d: gfx.Image | None = None
        self.material_2d: ImageBlockMaterial | None = None
        self._proxy_tex_2d: gfx.Texture | None = None
        self._aabb_line_2d: gfx.Line | None = None
        if "2d" in render_modes and image_geometry_2d is not None:
            inner, self.material_2d, self._proxy_tex_2d = self._build_2d_node(
                colormap=colormap, clim=clim, interpolation=interpolation
            )
            self._inner_node_2d = inner
            self.node_2d = gfx.Group()
            self.node_2d.add(inner)
            # Build AABB line with known geometry (geometry available at construction).
            self._aabb_line_2d = self._build_aabb_line_2d()
            self.node_2d.add(self._aabb_line_2d)

        if self.node_3d is not None:
            self.node_3d.render_order = render_order
        if self.node_2d is not None:
            self.node_2d.render_order = render_order

        # Apply node matrices now if displayed_axes are already known.
        # Without this, the matrices stay at identity until the first
        # displayed-axes change, which may never happen in a fixed viewer.
        if self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)

    @classmethod
    def from_cellier_model(
        cls,
        model: MultiscaleImageVisual,
        level_shapes: list[tuple[int, ...]],
        render_modes: set[str],
        displayed_axes: tuple[int, ...],
    ) -> GFXMultiscaleImageVisual:
        """Build a ``GFXMultiscaleImageVisual`` from a ``MultiscaleImageVisual`` model.

        All render-layer parameters (block size, GPU budgets, interpolation,
        brick shader flag) are read from ``model.render_config``.

        Parameters
        ----------
        model : MultiscaleImageVisual
            Source visual model.  Must have a ``render_config`` field.
        level_shapes : list[tuple[int, ...]]
            Full nD shape per level, finest first.
        render_modes : set[str]
            Which nodes to build: ``{"3d"}``, ``{"2d"}``, or ``{"2d", "3d"}``.
        displayed_axes : tuple[int, ...]
            The axes currently displayed.  ``len == 3`` means the initial mode
            is 3D; ``len == 2`` means 2D.  In dual-mode (``{"2d", "3d"}``),
            when starting in 3D the 2D geometry is built from
            ``displayed_axes[-2:]``.

        Returns
        -------
        GFXMultiscaleImageVisual
        """
        render_config = model.render_config
        block_size = render_config.block_size
        interpolation = render_config.interpolation
        use_brick_shader = render_config.use_brick_shader
        gpu_budget_bytes = render_config.gpu_budget_bytes
        gpu_budget_bytes_2d = render_config.gpu_budget_bytes_2d
        threshold = model.appearance.iso_threshold

        if len(displayed_axes) == 3:
            axes_3d: tuple[int, ...] | None = displayed_axes
            axes_2d: tuple[int, ...] = displayed_axes[-2:]
        else:
            axes_3d = None
            axes_2d = displayed_axes

        # Build 3D geometry when 3D rendering is requested and axes are available.
        volume_geometry: VolumeGeometry | None = None
        if "3d" in render_modes and axes_3d is not None:
            shapes_3d = [tuple(s[ax] for ax in axes_3d) for s in level_shapes]
            transforms_3d = [t.set_slice(axes_3d) for t in model.level_transforms]
            volume_geometry = VolumeGeometry(
                level_shapes=shapes_3d,
                level_transforms=transforms_3d,
                block_size=block_size,
            )

        # Build 2D geometry when 2D rendering is requested.
        image_geometry_2d: ImageGeometry2D | None = None
        if "2d" in render_modes:
            shapes_2d_full = [tuple(s[ax] for ax in axes_2d) for s in level_shapes]
            transforms_2d = [t.set_slice(axes_2d) for t in model.level_transforms]
            level_shapes_2d = [(s[0], s[1]) for s in shapes_2d_full]
            image_geometry_2d = ImageGeometry2D(
                level_shapes=level_shapes_2d,
                block_size=block_size,
                n_levels=len(level_shapes),
                level_transforms=transforms_2d,
            )

        colormap = model.appearance.color_map.to_pygfx(N=256)
        clim = model.appearance.clim

        instance = cls(
            visual_model_id=model.id,
            volume_geometry=volume_geometry,
            image_geometry_2d=image_geometry_2d,
            render_modes=render_modes,
            displayed_axes=displayed_axes,
            colormap=colormap,
            clim=clim,
            threshold=threshold,
            interpolation=interpolation,
            gpu_budget_bytes_3d=gpu_budget_bytes,
            gpu_budget_bytes_2d=gpu_budget_bytes_2d,
            aabb_enabled=model.aabb.enabled,
            aabb_color=model.aabb.color,
            aabb_line_width=model.aabb.line_width,
            render_order=model.appearance.render_order,
            transform=model.transform,
            full_level_transforms=list(model.level_transforms),
            full_level_shapes=list(level_shapes),
            use_brick_shader=use_brick_shader,
        )
        for mat in (instance.material_3d, instance.material_2d):
            if mat is not None:
                mat.depth_test = model.appearance.depth_test
                mat.depth_write = model.appearance.depth_write
                mat.depth_compare = model.appearance.depth_compare
        return instance

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def n_levels(self) -> int:
        """Number of LOD levels."""
        if self._volume_geometry is not None:
            return self._volume_geometry.n_levels
        if self._image_geometry_2d is not None:
            return self._image_geometry_2d.n_levels
        raise RuntimeError("No geometry available")

    # ── Node selection ───────────────────────────────────────────────

    def get_node_for_dims(self, displayed_axes: tuple[int, ...]) -> gfx.Group | None:
        """Rebuild geometry if needed and return the node for *displayed_axes*.

        Calls ``self.rebuild_geometry`` internally using the already-stored
        ``_full_level_shapes``.  The controller no longer needs to supply
        level shapes or call ``rebuild_geometry`` directly.

        Parameters
        ----------
        displayed_axes : tuple[int, ...]
            The new set of displayed axes.

        Returns
        -------
        gfx.Group or None
            The new active node after geometry rebuild, or ``None`` if the
            required render mode was not built.
        """
        _old_node, new_node = self.rebuild_geometry(
            self._full_level_shapes, displayed_axes
        )
        return new_node

    # ── Geometry rebuild ─────────────────────────────────────────────

    def rebuild_geometry(
        self,
        level_shapes: list[tuple[int, ...]],
        displayed_axes: tuple[int, ...],
    ) -> tuple[gfx.WorldObject | None, gfx.WorldObject | None]:
        """Rebuild geometry and GPU resources after a displayed_axes change.

        Returns ``(old_node, new_node)`` for the active node so the caller
        can swap it in the scene graph.  Derives the 3D and 2D axis subsets
        from ``ndim`` (last-three and last-two axes respectively).

        Parameters
        ----------
        level_shapes : list[tuple[int, ...]]
            Full nD shape per level from the data store.
        displayed_axes : tuple[int, ...]
            The new set of displayed axes.  ``len == 3`` rebuilds the 3D
            geometry using exactly those axes; ``len == 2`` rebuilds the 2D
            geometry.  Also determines which node is the active one to swap.

        Returns
        -------
        tuple[old_node, new_node]
            The previous and replacement pygfx nodes (may be ``None``).
        """
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
        """Rebuild 3D GPU resources after geometry update."""
        geo = self._volume_geometry
        # Clear cache
        self._block_cache_3d.tile_manager.release_all_in_flight()
        # Rebuild LUT manager
        self._lut_manager_3d = LutIndirectionManager3D(
            base_layout=geo.base_layout,
            n_levels=geo.n_levels,
            level_scale_vecs_data=geo._scale_vecs_data,
        )
        # Rebuild param buffers
        self._lut_params_buffer_3d = build_lut_params_buffer_3d(
            geo.base_layout,
            self._block_cache_3d.info,
            proxy_voxels_per_brick=geo.block_size,
        )
        self._block_scales_buffer_3d = build_block_scales_buffer_3d(
            geo._scale_vecs_data
        )
        # Rebuild node preserving current appearance
        if self.node_3d is not None:
            colormap = self.material_3d.map
            clim = self.material_3d.clim
            threshold = self.material_3d.threshold
            interpolation = self.material_3d.interpolation
            inner, self.material_3d, self._proxy_tex_3d = self._build_3d_node(  # type: ignore[assignment]
                colormap=colormap,
                clim=clim,
                threshold=threshold,
                interpolation=interpolation,
            )
            self._inner_node_3d = inner
            self._aabb_line_3d = self._build_aabb_line_3d()
            self.node_3d = gfx.Group()
            self.node_3d.add(inner)
            self.node_3d.add(self._aabb_line_3d)
            if self._last_displayed_axes is not None:
                self._update_node_matrix(self._last_displayed_axes)
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
            scale_vecs_data=geo2d._scale_vecs_data,
        )
        # Rebuild param buffers
        self._lut_params_buffer_2d = build_lut_params_buffer_2d(
            geo2d.base_layout, self._block_cache_2d.info
        )
        self._block_scales_buffer_2d = build_block_scales_buffer_2d(
            level_scale_vecs_data=geo2d._scale_vecs_data,
        )
        # Rebuild node preserving current appearance
        if self.node_2d is not None:
            colormap = self.material_2d.map
            clim = self.material_2d.clim
            interpolation = self.material_2d.interpolation
            inner, self.material_2d, self._proxy_tex_2d = self._build_2d_node(
                colormap=colormap, clim=clim, interpolation=interpolation
            )
            self._inner_node_2d = inner
            self._aabb_line_2d = self._build_aabb_line_2d()
            self.node_2d = gfx.Group()
            self.node_2d.add(inner)
            self.node_2d.add(self._aabb_line_2d)
            if self._last_displayed_axes is not None:
                self._update_node_matrix(self._last_displayed_axes)
        self._pending_slot_map_2d = {}

    def _update_node_matrix(self, displayed_axes: tuple[int, ...]) -> None:
        """Recompute and apply the pygfx node matrix for *displayed_axes*."""
        self._last_displayed_axes = displayed_axes
        sub = self._transform.set_slice(displayed_axes)

        if self._use_brick_shader and self.node_3d is not None:
            # Brick shader: compose normalized → data → world.
            data_to_world = _pygfx_matrix(sub)
            m = compose_world_transform(
                data_to_world, self._dataset_size, self._norm_size
            )
            self.node_3d.local.matrix = m
        elif self.node_3d is not None:
            m = _pygfx_matrix(sub)
            self.node_3d.local.matrix = m

        if self.node_2d is not None:
            self.node_2d.local.matrix = _pygfx_matrix(sub)

    def _build_world_to_level_transforms(
        self,
    ) -> list[AffineTransform]:
        """Precompute composed world→level-k transforms.

        All transforms operate in data-axis order (z, y, x).
        ``map_coordinates`` on a world-space point gives the level-k
        voxel index.

        Composition: ``inv_level @ inv_visual`` so that
        ``map_coordinates(world_pt)`` applies inv_visual first
        (world → level-0), then inv_level (level-0 → level-k).
        """
        inv_visual = AffineTransform(matrix=self._transform.inverse_matrix)
        result: list[AffineTransform] = []
        for lt in self._level_transforms:
            inv_level = AffineTransform(matrix=lt.inverse_matrix)
            # world → level-0 → level-k
            composed = inv_level @ inv_visual
            result.append(composed)
        return result

    # ── 3D SliceCoordinator interface ──────────────────────────────────

    def build_slice_request(
        self,
        camera_pos_world: np.ndarray,
        frustum_corners_world: np.ndarray | None,
        thresholds: list[float] | None,
        dims_state: DimsState | None = None,
        force_level: int | None = None,
    ) -> list[ChunkRequest]:
        """Run the synchronous 3D planning phase and return ChunkRequests.

        World-space inputs are transformed to data space before planning.

        Pipeline: LOAD select -> distance sort -> optional frustum cull ->
        budget cap -> stage() -> build ``ChunkRequest`` objects.

        Parameters
        ----------
        camera_pos_world : np.ndarray
            Camera position in world coordinates.
        frustum_corners_world : np.ndarray or None
            Frustum corner points in world coordinates. ``None`` disables
            frustum culling.
        thresholds : list[float] or None
            LOD distance thresholds.
        dims_state : DimsState or None
            Current dimension state.
        force_level : int or None
            Override LOD level.

        Returns
        -------
        list[ChunkRequest]
        """
        t_plan_start = time.perf_counter()
        self._frame_number += 1
        geo = self._volume_geometry

        # Lazy node matrix update when displayed axes change.
        if dims_state is not None:
            displayed = dims_state.selection.displayed_axes
            if displayed != self._last_displayed_axes:
                self._update_node_matrix(displayed)

        # Camera / frustum are always 3D from pygfx -- use the 3D
        # sub-transform (last 3 displayed axes) for inverse mapping.
        sub_3d = self._transform.set_slice(
            self._last_displayed_axes or tuple(range(self._ndim))[-3:]
        )
        # level_grids["centres"], bricks_in_frustum_arr, and sort_arr_by_distance
        # all work in shader order (x=W, y=H, z=D) level-0 voxel space.
        # sub_3d (AffineTransform) operates in data-axis order (z,y,x), whereas
        # pygfx world coords are in (x,y,z). Reverse axes before imap and back
        # after to land in the correct (x,y,z) voxel space.
        cam_zyx = camera_pos_world[[2, 1, 0]]
        cam_data_zyx = sub_3d.imap_coordinates(cam_zyx.reshape(1, -1)).flatten()
        camera_pos_data = cam_data_zyx[[2, 1, 0]]  # back to shader (x,y,z)

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

        # Slice indices are mapped per-level via composed transforms
        # (world → level-k) inside _build_axis_selections.
        # 1. LOAD selection
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

        # Log min/max brick-centre distances at the finest level for diagnosis.
        finest_centres = geo._level_grids[0]["centres"]  # (M, 3) voxel space
        if len(finest_centres):
            cam = np.asarray(camera_pos_data, dtype=np.float64)
            dists = np.sqrt(((finest_centres - cam) ** 2).sum(axis=1))
            _PERF_LOGGER.debug(
                "[frame %d]  brick_dist  min=%.1f  max=%.1f  median=%.1f"
                "  (finest level, voxel space)",
                self._frame_number,
                dists.min(),
                dists.max(),
                float(np.median(dists)),
            )

        # 2. Distance sort — brick centres and camera are both in level-0 voxel space.
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

        # 3. Frustum cull
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

        # Log per-level breakdown of what was requested this frame.
        if len(brick_arr):
            level_col = brick_arr[:, 0]
            requested_by_level = {
                int(lv): int((level_col == lv).sum()) for lv in np.unique(level_col)
            }
        else:
            requested_by_level = {}
        _PERF_LOGGER.debug(
            "[frame %d]  requested_by_level=%s  thresholds=%s",
            self._frame_number,
            requested_by_level,
            [f"{t:.1f}" for t in (thresholds or [])],
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
            slot.brick_max = float(data.max())
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

        # On first brick batch, reveal the AABB line if enabled.
        if not self._data_ready_3d and self._aabb_line_3d is not None:
            self._data_ready_3d = True
            self._aabb_line_3d.visible = self._aabb_enabled

    def cancel_pending(self) -> None:
        """Release all in-flight 3D slots."""
        if self._block_cache_3d is None:
            return
        self._block_cache_3d.tile_manager.release_all_in_flight()
        self._pending_slot_map = {}

    # ── 2D SliceCoordinator interface ──────────────────────────────────

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
        """Run the synchronous 2D planning phase and return ChunkRequests.

        World-space inputs are transformed to data space before planning.

        Pipeline: LOD select -> distance sort -> optional viewport cull ->
        budget cap -> stage() -> build ``ChunkRequest`` objects.

        Parameters
        ----------
        camera_pos_world : ndarray, shape (3,)
            Camera world-space position ``(x, y, z)``.
        viewport_width_px : float
            Viewport width in logical pixels.
        world_width : float
            Visible world width in world units.
        view_min_world : ndarray, shape (2,) or None
            Viewport AABB minimum ``(x, y)`` in world space.
            ``None`` disables culling.
        view_max_world : ndarray, shape (2,) or None
            Viewport AABB maximum ``(x, y)`` in world space.
            ``None`` disables culling.
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

        # Record the current slice coordinate for two-phase LUT rebuild (Step 3).
        self._current_slice_coord = tuple(
            sorted(dims_state.selection.slice_indices.items())
        )

        geo2d = self._image_geometry_2d
        block_size = geo2d.block_size
        n_levels = geo2d.n_levels

        # Lazy node matrix update when displayed axes change.
        displayed = dims_state.selection.displayed_axes
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)

        # Camera / viewport are 2D from pygfx -- use the 2D
        # sub-transform for inverse mapping.
        #
        # Axis-order note: pygfx delivers world coords as (X, Y) where
        #   X = second displayed axis,  Y = first displayed axis.
        # But set_slice(displayed=(a, b)).imap_coordinates expects input in
        #   (a-world, b-world) = (first-displayed, second-displayed) order
        # and returns (a-data, b-data) = (gy, gx) order.
        # Downstream consumers (sort_tiles_by_distance_2d, viewport_cull_2d)
        # expect (gx, gy) = (second-displayed, first-displayed) order.
        # So we swap the input and swap the output.
        sub_2d = self._transform.set_slice(displayed)
        # camera_pos_world[:2] is (X_world, Y_world); imap expects (Y_world, X_world).
        camera_pos_2d = sub_2d.imap_coordinates(
            camera_pos_world[[1, 0]].reshape(1, -1)
        ).flatten()[[1, 0]]  # swap output (gy, gx) -> (gx, gy)
        # Pad to 3D for compatibility with downstream code.
        camera_pos = np.array(
            [camera_pos_2d[0], camera_pos_2d[1], 0.0], dtype=np.float32
        )

        # Transform viewport AABB to data space via corners.
        if use_culling and view_min_world is not None and view_max_world is not None:
            cx = float(camera_pos_world[0])  # X_world = second-displayed axis
            cy = float(camera_pos_world[1])  # Y_world = first-displayed axis
            half_w = world_width / 2.0
            half_h = (float(view_max_world[1]) - float(view_min_world[1])) / 2.0
            # imap expects (first-displayed, second-displayed) = (cy, cx) order.
            corners_world_2d = np.array(
                [
                    [cy - half_h, cx - half_w],
                    [cy - half_h, cx + half_w],
                    [cy + half_h, cx + half_w],
                    [cy + half_h, cx - half_w],
                ],
                dtype=np.float32,
            )
            # imap returns (gy, gx); swap columns to (gx, gy) for culling.
            corners_data_2d = sub_2d.imap_coordinates(corners_world_2d)[:, [1, 0]]
            view_min = corners_data_2d.min(axis=0)
            view_max = corners_data_2d.max(axis=0)
        else:
            view_min = None
            view_max = None

        # ── DEBUG: print camera & viewport in level-0 voxel space ──────────
        import logging as _logging

        _DBG = _logging.getLogger("cellier.2d_reslice_debug")
        if _DBG.isEnabledFor(_logging.DEBUG):
            _DBG.debug(
                "[2D plan] displayed=%s  camera_L0_vox(gx,gy)=(%.2f, %.2f)  "
                "viewport_L0_vox: min=%s  max=%s",
                displayed,
                float(camera_pos[0]),
                float(camera_pos[1]),
                (
                    f"({view_min[0]:.2f},{view_min[1]:.2f})"
                    if view_min is not None
                    else "None"
                ),
                (
                    f"({view_max[0]:.2f},{view_max[1]:.2f})"
                    if view_max is not None
                    else "None"
                ),
            )

        # Slice indices mapped per-level via composed transforms.

        # 1. LOD selection
        t0 = time.perf_counter()
        tile_arr = select_lod_2d(
            geo2d._level_grids,
            n_levels,
            viewport_width_px=viewport_width_px,
            world_width=world_width,
            lod_bias=lod_bias,
            force_level=force_level,
            level_scale_factors=geo2d._level_scale_factors,
        )
        lod_select_ms = (time.perf_counter() - t0) * 1000

        # 2. Distance sort
        t0 = time.perf_counter()
        tile_arr = sort_tiles_by_distance_2d(
            tile_arr,
            camera_pos,
            block_size,
            level_scale_arr_shader=geo2d._scale_arr_shader,
            level_translation_arr_shader=geo2d._translation_arr_shader,
        )
        distance_sort_ms = (time.perf_counter() - t0) * 1000

        # Convert to dict (embed current slice coord into every key).
        required = arr_to_block_keys_2d(tile_arr, slice_coord=self._current_slice_coord)
        n_total = len(required)

        # 3. Viewport culling
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

        # 4. Budget truncation
        n_needed = len(required)
        n_budget = self._block_cache_2d.info.n_slots - 1
        n_dropped = max(0, n_needed - n_budget)
        if n_dropped:
            keys_to_keep = list(required.keys())[:n_budget]
            required = {k: required[k] for k in keys_to_keep}

        # 5. Evict finer-than-target tiles, then stage.
        # Evicting first returns slots to free_slots so the incoming coarser
        # tiles can claim them without triggering unnecessary LRU evictions.
        target_level = int(tile_arr[0, 0]) if len(tile_arr) > 0 else 1

        n_evicted = self._block_cache_2d.tile_manager.evict_finer_than(target_level)

        t0 = time.perf_counter()
        fill_plan = self._block_cache_2d.tile_manager.stage(
            required, self._frame_number
        )
        stage_ms = (time.perf_counter() - t0) * 1000

        # When tiles were evicted but all required coarse tiles are cache hits,
        # on_data_ready_2d never fires so the LUT would remain stale (still
        # pointing to freed slots).  Rebuild immediately in that case.
        if n_evicted > 0 and not fill_plan:
            self._lut_manager_2d.rebuild(
                self._block_cache_2d.tile_manager,
                current_slice_coord=self._current_slice_coord,
            )

        # 6. Build ChunkRequests
        slice_id = uuid4()
        chunk_requests: list[ChunkRequest] = []
        self._pending_slot_map_2d = {}

        overlap = self._block_cache_2d.info.overlap

        ndim = len(dims_state.axis_labels)
        sel = dims_state.selection

        _debug_tile_count = 0
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

            # ── DEBUG: print tile region and data selection ─────────────────
            if _DBG.isEnabledFor(_logging.DEBUG) and _debug_tile_count < 4:
                _sv = geo2d._scale_arr_shader[level_index]
                _cx_l0 = (tile_key.gx + 0.5) * block_size * float(_sv[0])
                _cy_l0 = (tile_key.gy + 0.5) * block_size * float(_sv[1])
                _DBG.debug(
                    "  tile level=%d gy=%d gx=%d  "
                    "centre_L0_vox(gx,gy)=(%.2f,%.2f)  axis_sel=%s",
                    tile_key.level,
                    tile_key.gy,
                    tile_key.gx,
                    _cx_l0,
                    _cy_l0,
                    axis_selections,
                )
                _debug_tile_count += 1

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

        self._lut_manager_2d.rebuild(
            self._block_cache_2d.tile_manager,
            current_slice_coord=self._current_slice_coord,
        )

        _GPU_LOGGER.info(
            "lut_rebuilt  resident=%d  frame=%d",
            n_resident,
            self._frame_number,
        )

        # On first tile batch, reveal the AABB line if enabled.
        if not self._data_ready_2d and self._aabb_line_2d is not None:
            self._data_ready_2d = True
            self._aabb_line_2d.visible = self._aabb_enabled

    def cancel_pending_2d(self) -> None:
        """Release all in-flight 2D slots."""
        if self._block_cache_2d is None:
            return
        self._block_cache_2d.tile_manager.release_all_in_flight()
        self._pending_slot_map_2d = {}

    def invalidate_2d_cache(self) -> None:
        """Cancel in-flight 2D requests when the slice position changes.

        Old committed tiles are intentionally kept alive as a visible fallback
        while new tiles load.  Because ``BlockKey2D`` now encodes
        ``slice_coord``, tiles from different slice positions cannot collide, so
        keeping them does not cause rendering artefacts.

        A full LUT rebuild is not needed here — the LUT remains current from
        the last ``on_data_ready_2d`` call and old tiles are the correct thing
        to display until new ones arrive.
        """
        if self._block_cache_2d is None or self._lut_manager_2d is None:
            return
        self.cancel_pending_2d()

    # ── EventBus handler methods ─────────────────────────────────────────

    def on_transform_changed(self, event: TransformChangedEvent) -> None:
        """Update stored transform, norm_size uniform, and pygfx node matrix."""
        self._transform = event.transform
        self._world_to_level_transforms = self._build_world_to_level_transforms()

        # Recompute norm_size and push the updated uniform buffer so the
        # proxy cube shape stays consistent with the new transform.
        if self._use_brick_shader and self._dataset_size is not None:
            _check_transform_no_rotation(self._transform)
            self._norm_size = _norm_size_from_transform(
                self._transform, self._dataset_size
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
        elif event.field_name == "iso_threshold":
            if self.material_3d is not None:
                self.material_3d.threshold = float(event.new_value)
        elif event.field_name == "render_mode":
            if self.material_3d is not None:
                self.material_3d.render_mode = event.new_value
        elif event.field_name == "render_order":
            if self.node_3d is not None:
                self.node_3d.render_order = event.new_value
            if self.node_2d is not None:
                self.node_2d.render_order = event.new_value
        elif event.field_name in ("depth_test", "depth_write", "depth_compare"):
            for mat in (self.material_3d, self.material_2d):
                if mat is not None:
                    setattr(mat, event.field_name, event.new_value)

    def on_visibility_changed(self, event: VisualVisibilityChangedEvent) -> None:
        """Apply visibility change to all render nodes."""
        if self.node_3d is not None:
            self.node_3d.visible = event.visible
        if self.node_2d is not None:
            self.node_2d.visible = event.visible

    def on_aabb_changed(self, event: AABBChangedEvent) -> None:
        """Apply an AABB parameter change.

        ``enabled`` toggles AABB line visibility (guarded by data-ready
        flags so the line cannot appear before the first brick/tile batch).
        ``color`` updates the line material.

        Parameters
        ----------
        event : AABBChangedEvent
            Carries ``field_name`` and ``new_value``.
        """
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
        """Stub — brick eviction deferred to a future phase."""
        pass

    def on_data_store_metadata_changed(
        self, event: DataStoreMetadataChangedEvent
    ) -> None:
        """Stub — geometry rebuild deferred to a future phase."""
        pass

    def tick(self) -> None:
        """Advance jitter seed for the brick shader. No-op otherwise."""
        if (
            self._use_brick_shader
            and self.material_3d is not None
            and isinstance(self.material_3d, MultiscaleVolumeBrickMaterial)
        ):
            self.material_3d.tick()

    # ── Private helpers ─────────────────────────────────────────────────

    def _build_aabb_line_3d(self) -> gfx.Line:
        """Build the 3D AABB wireframe line for the current geometry.

        For the brick-shader path the line is in normalized space
        (matching the Group matrix).  For the standard block shader the
        line is in data-voxel space.
        """
        if self._use_brick_shader and self._norm_size is not None:
            half = self._norm_size / 2.0
            positions = _box_wireframe_positions(-half, half)
        elif self._volume_geometry is not None:
            d, h, w = self._volume_geometry.level_shapes[0]
            # pygfx voxel convention: voxel i center at i, edges at i±0.5.
            # The inner Volume's local transform (scale=bs, offset=bs/2-0.5)
            # maps data voxel n to Group-local n, so the extent is [-0.5, N-0.5].
            positions = _box_wireframe_positions(
                np.array([-0.5, -0.5, -0.5]),
                np.array([w - 0.5, h - 0.5, d - 0.5]),
            )
        else:
            positions = _box_wireframe_positions(np.zeros(3), np.ones(3))
        return _make_aabb_line(positions, self._aabb_color, self._aabb_line_width)

    def _build_aabb_line_2d(self) -> gfx.Line:
        """Build the 2D AABB wireframe rect for the current geometry."""
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

    def _build_3d_node_brick(
        self,
        colormap: gfx.TextureMap,
        clim: tuple[float, float],
        threshold: float,
    ) -> tuple[gfx.Volume, MultiscaleVolumeBrickMaterial, gfx.Texture]:
        """Construct the proxy texture, brick material, and Volume node.

        The brick shader generates its own box geometry from
        ``u_vol_params.norm_size_*``, so the proxy texture is a small
        dummy (2x2x2) and the inner Volume node has identity local
        transform.  The Group node's matrix (set by
        ``_update_node_matrix``) maps normalized -> world.
        """
        proxy_data = np.zeros((2, 2, 2), dtype=np.float32)
        proxy_tex = gfx.Texture(proxy_data, dim=3)

        material = MultiscaleVolumeBrickMaterial(
            cache_texture=self._block_cache_3d.cache_tex,
            lut_texture=self._lut_manager_3d.lut_tex,
            brick_max_texture=self._lut_manager_3d.brick_max_tex,
            vol_params_buffer=self._vol_params_buffer,
            block_scales_buffer=self._brick_scales_buffer,
            clim=clim,
            map=colormap,
            threshold=threshold,
        )

        geometry = gfx.Geometry(grid=proxy_tex)
        vol = gfx.Volume(geometry, material)
        # No inner transform — vertex shader uses normalized space.

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
        h, w = self._image_geometry_2d.level_shapes[0]
        # Proxy grid rounds up to whole tiles. Apply a per-axis correction so
        # the displayed quad footprint matches the true finest-level (W, H).
        sx = float(w) / float(gw * bs)
        sy = float(h) / float(gh * bs)
        scale_x = float(bs) * sx
        scale_y = float(bs) * sy
        image.local.scale = (scale_x, scale_y, 1.0)
        image.local.position = (scale_x * 0.5, scale_y * 0.5, 0.0)

        return image, material, proxy_tex
