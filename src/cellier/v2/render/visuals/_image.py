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
from cellier.v2.render.visuals._image_memory import (
    _pygfx_matrix,
)
from cellier.v2.transform import AffineTransform

if TYPE_CHECKING:
    from cellier.v2._state import AxisAlignedSelectionState, DimsState
    from cellier.v2.events._events import (
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

# Importing this module registers VolumeBlockShader with pygfx via the
# @register_wgpu_render_function decorator.
import cellier.v2.render.shaders._block_volume as _shader_reg  # noqa: F401

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
        transform: AffineTransform | None = None,
        full_level_transforms: list[AffineTransform] | None = None,
        full_level_shapes: list[tuple[int, ...]] | None = None,
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
        # Track displayed axes for lazy node matrix updates (Option C).
        self._last_displayed_axes: tuple[int, ...] | None = None
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
                volume_geometry._scale_vecs_data
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
                level_scale_vecs_data=image_geometry_2d._scale_vecs_data,
            )

        if colormap is None:
            colormap = gfx.cm.viridis

        # ── 3D node ─────────────────────────────────────────────────────
        self.node_3d: gfx.Group | None = None
        self._inner_node_3d: gfx.Volume | None = None
        self.material_3d: VolumeBlockMaterial | None = None
        self._proxy_tex_3d: gfx.Texture | None = None
        if "3d" in render_modes and volume_geometry is not None:
            inner, self.material_3d, self._proxy_tex_3d = self._build_3d_node(
                colormap=colormap,
                clim=clim,
                threshold=threshold,
                interpolation=interpolation,
            )
            self._inner_node_3d = inner
            self.node_3d = gfx.Group()
            self.node_3d.add(inner)
            # Node matrix set lazily on first build_slice_request.

        # ── 2D node ─────────────────────────────────────────────────────
        self.node_2d: gfx.Group | None = None
        self._inner_node_2d: gfx.Image | None = None
        self.material_2d: ImageBlockMaterial | None = None
        self._proxy_tex_2d: gfx.Texture | None = None
        if "2d" in render_modes and image_geometry_2d is not None:
            inner, self.material_2d, self._proxy_tex_2d = self._build_2d_node(
                colormap=colormap, clim=clim, interpolation=interpolation
            )
            self._inner_node_2d = inner
            self.node_2d = gfx.Group()
            self.node_2d.add(inner)
            # Node matrix set lazily on first build_slice_request_2d.

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

        # Extract displayed-subspace transforms for geometry objects.
        displayed_transforms = [
            t.set_slice(displayed_axes) for t in model.level_transforms
        ]

        # Build 3D geometry only when 3D rendering is requested.
        volume_geometry: VolumeGeometry | None = None
        if "3d" in render_modes and len(displayed_axes) == 3:
            volume_geometry = VolumeGeometry(
                level_shapes=displayed_level_shapes,
                level_transforms=displayed_transforms,
                block_size=block_size,
            )

        # Build 2D geometry only when 2D rendering is requested.
        image_geometry_2d: ImageGeometry2D | None = None
        if "2d" in render_modes and len(displayed_axes) == 2:
            level_shapes_2d = [(s[0], s[1]) for s in displayed_level_shapes]
            image_geometry_2d = ImageGeometry2D(
                level_shapes=level_shapes_2d,
                block_size=block_size,
                n_levels=len(level_shapes),
                level_transforms=displayed_transforms,
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
            transform=model.transform,
            full_level_transforms=list(model.level_transforms),
            full_level_shapes=list(level_shapes),
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
        # Update full-ndim shapes (data fetching).
        self._full_level_shapes = list(level_shapes)

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
            geo._scale_vecs_data
        )
        # Rebuild node preserving current appearance
        if self.node_3d is not None:
            colormap = self.material_3d.map
            clim = self.material_3d.clim
            threshold = self.material_3d.threshold
            interpolation = self.material_3d.interpolation
            inner, self.material_3d, self._proxy_tex_3d = self._build_3d_node(
                colormap=colormap,
                clim=clim,
                threshold=threshold,
                interpolation=interpolation,
            )
            self._inner_node_3d = inner
            self.node_3d = gfx.Group()
            self.node_3d.add(inner)
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
            self.node_2d = gfx.Group()
            self.node_2d.add(inner)
            if self._last_displayed_axes is not None:
                self._update_node_matrix(self._last_displayed_axes)
        self._pending_slot_map_2d = {}

    # ── Node matrix update (Option C -- lazy, displayed-axes-aware) ──

    def _update_node_matrix(self, displayed_axes: tuple[int, ...]) -> None:
        """Recompute and apply the pygfx node matrix for *displayed_axes*."""
        self._last_displayed_axes = displayed_axes
        sub = self._transform.set_slice(displayed_axes)
        m = _pygfx_matrix(sub)
        if self.node_3d is not None:
            self.node_3d.local.matrix = m
        if self.node_2d is not None:
            self.node_2d.local.matrix = m

    # ── Composed world→level-k transforms ───────────────────────────────

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
        camera_pos = sub_3d.imap_coordinates(camera_pos_world.reshape(1, -1)).flatten()

        if frustum_corners_world is not None:
            corners_data = sub_3d.imap_coordinates(
                frustum_corners_world.reshape(-1, 3)
            ).reshape(frustum_corners_world.shape)
            frustum_planes = frustum_planes_from_corners(corners_data)
        else:
            frustum_planes = None

        # Slice indices are mapped per-level via composed transforms
        # (world → level-k) inside _build_axis_selections.
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
        geo2d = self._image_geometry_2d
        block_size = geo2d.block_size
        n_levels = geo2d.n_levels

        # Lazy node matrix update when displayed axes change.
        displayed = dims_state.selection.displayed_axes
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)

        # Camera / viewport are 2D from pygfx -- use the 2D
        # sub-transform for inverse mapping.
        sub_2d = self._transform.set_slice(displayed)
        camera_pos_2d = sub_2d.imap_coordinates(
            camera_pos_world[:2].reshape(1, -1)
        ).flatten()
        # Pad to 3D for compatibility with downstream code.
        camera_pos = np.array(
            [camera_pos_2d[0], camera_pos_2d[1], 0.0], dtype=np.float32
        )

        # Transform viewport AABB to data space via corners.
        if use_culling and view_min_world is not None and view_max_world is not None:
            cx = float(camera_pos_world[0])
            cy = float(camera_pos_world[1])
            half_w = world_width / 2.0
            half_h = (float(view_max_world[1]) - float(view_min_world[1])) / 2.0
            corners_world_2d = np.array(
                [
                    [cx - half_w, cy - half_h],
                    [cx + half_w, cy - half_h],
                    [cx + half_w, cy + half_h],
                    [cx - half_w, cy + half_h],
                ],
                dtype=np.float32,
            )
            corners_data_2d = sub_2d.imap_coordinates(corners_world_2d)
            view_min = corners_data_2d.min(axis=0)
            view_max = corners_data_2d.max(axis=0)
        else:
            view_min = None
            view_max = None

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

        # Convert to dict
        required = arr_to_block_keys_2d(tile_arr)
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

    def on_transform_changed(self, event: TransformChangedEvent) -> None:
        """Update stored transform and pygfx node matrix."""
        self._transform = event.transform
        self._world_to_level_transforms = self._build_world_to_level_transforms()
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
