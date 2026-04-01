# src/cellier/v2/render/visuals/_image_memory.py
from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx

from cellier.v2._state import AxisAlignedSelectionState, DimsState
from cellier.v2.data.image._image_requests import ChunkRequest

if TYPE_CHECKING:
    from cellier.v2.data.image._image_memory_store import ImageMemoryStore
    from cellier.v2.events._events import (
        AppearanceChangedEvent,
        TransformChangedEvent,
        VisualVisibilityChangedEvent,
    )
    from cellier.v2.transform import AffineTransform
    from cellier.v2.visuals._image_memory import ImageVisual


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_colormap(color_map) -> gfx.TextureMap:
    """Convert a cmap Colormap to a pygfx TextureMap via cmap's pygfx bridge."""
    return color_map.to_pygfx(N=256)


def _pygfx_matrix(transform: AffineTransform) -> np.ndarray:
    """Embed a 2-D or 3-D AffineTransform into a pygfx-compatible 4x4 matrix.

    pygfx always requires a 4x4 matrix for ``node.local.matrix``.  This
    function converts from data-axis order to pygfx/shader order and
    places the transform into the correct positions of a 4x4 identity
    matrix.

    Data-axis order is ``(z, y, x)`` for 3D and ``(y, x)`` for 2D.
    pygfx uses ``(x, y, z)`` order, so the spatial axes are reversed.

    Parameters
    ----------
    transform : AffineTransform
        A transform with ``ndim`` in {1, 2, 3}.

    Returns
    -------
    np.ndarray
        A ``(4, 4)`` float32 matrix.
    """
    nd = transform.ndim
    src = transform.matrix
    # Reverse axis order: data (z, y, x) → pygfx (x, y, z).
    swap = list(reversed(range(nd)))
    m = np.eye(4, dtype=np.float32)
    for dst_i, src_i in enumerate(swap):
        for dst_j, src_j in enumerate(swap):
            m[dst_i, dst_j] = src[src_i, src_j]
        m[dst_i, 3] = src[src_i, nd]
    return m


def _transform_slice_indices(
    slice_indices: dict[int, int],
    transform: AffineTransform,
    store_shape: tuple[int, ...],
) -> dict[int, int]:
    """Map world-space slice positions to data-space voxel indices.

    The transform is N-dimensional and covers every data axis.  Each
    world-space slice position is placed into an N-D point, the inverse
    transform is applied, and the result is rounded and clamped to valid
    voxel indices.

    Parameters
    ----------
    slice_indices : dict[int, int]
        Axis -> world-space slice position.
    transform : AffineTransform
        N-D data-to-world transform (its inverse maps world -> data).
    store_shape : tuple[int, ...]
        Shape of the data store, used for clamping.

    Returns
    -------
    dict[int, int]
        Axis -> data-space voxel index.
    """
    if not slice_indices:
        return slice_indices

    ndim = transform.ndim
    world_pt = np.zeros(ndim, dtype=np.float64)
    for data_axis, world_pos in slice_indices.items():
        world_pt[data_axis] = float(world_pos)

    data_pt = transform.imap_coordinates(world_pt.reshape(1, -1)).flatten()

    result: dict[int, int] = {}
    for data_axis in slice_indices:
        raw = float(data_pt[data_axis])
        idx = int(round(raw))
        idx = max(0, min(idx, store_shape[data_axis] - 1))
        result[data_axis] = idx
    return result


def _build_axis_selections(
    dims_state: DimsState,
    store_shape: tuple[int, ...],
) -> tuple[int | tuple[int, int], ...]:
    """Build axis_selections for a ChunkRequest from a DimsState.

    Displayed axes receive ``(0, store_shape[axis])`` -- the full extent.
    Sliced axes receive their integer slice index.

    Parameters
    ----------
    dims_state : DimsState
        Immutable snapshot from the controller.
    store_shape : tuple[int, ...]
        Shape of the backing numpy array (from ``ImageMemoryStore.shape``).

    Returns
    -------
    tuple[int | tuple[int, int], ...]
        One entry per data axis, in data axis order.
    """
    sel = dims_state.selection  # AxisAlignedSelectionState
    ndim = len(store_shape)
    result: list[int | tuple[int, int]] = []
    for ax in range(ndim):
        if ax in sel.displayed_axes:
            result.append((0, store_shape[ax]))
        else:
            result.append(sel.slice_indices[ax])
    return tuple(result)


# ---------------------------------------------------------------------------
# GFXImageMemoryVisual
# ---------------------------------------------------------------------------


class GFXImageMemoryVisual:
    """Render-layer visual for one ``ImageVisual`` backed by ``ImageMemoryStore``.

    Owns a single pygfx node -- either ``gfx.Image`` (render_mode=="2d") or
    ``gfx.Volume`` (render_mode=="3d") -- and a placeholder 1x1 or 2x2x2
    texture that is replaced on the first ``on_data_ready[_2d]`` call.

    There is no brick cache, no LUT indirection, and no LOD selection. Every
    reslice produces exactly one ``ChunkRequest`` for the full slice / volume.
    The node geometry is replaced on every commit.

    Parameters
    ----------
    visual_model : ImageVisual
        Associated model-layer visual. Provides the initial appearance.
    data_store : ImageMemoryStore
        The backing data store. Used to query shape in planning methods.
    render_mode : str
        ``"2d"`` builds a ``gfx.Image`` node; ``"3d"`` builds a
        ``gfx.Volume`` node.
    """

    def __init__(
        self,
        visual_model: ImageVisual,
        data_store: ImageMemoryStore,
        render_mode: str,
        transform: AffineTransform | None = None,
    ) -> None:
        if render_mode not in ("2d", "3d"):
            raise ValueError(f"render_mode must be '2d' or '3d', got {render_mode!r}")

        self.visual_model_id: UUID = visual_model.id
        self._render_mode = render_mode
        self._data_store = data_store

        # Store the data-to-world transform, auto-promoting if needed.
        if transform is None:
            from cellier.v2.transform import AffineTransform as _AT

            transform = _AT.identity(ndim=data_store.ndim)
        elif transform.ndim < data_store.ndim:
            transform = transform.expand_dims(data_store.ndim)
        self._transform: AffineTransform = transform

        # Track displayed axes for lazy node matrix updates (Option C).
        self._last_displayed_axes: tuple[int, ...] | None = None

        appearance = visual_model.appearance
        colormap = _make_colormap(appearance.color_map)

        if render_mode == "2d":
            # Placeholder 1x1 texture -- replaced on first on_data_ready_2d.
            placeholder = np.zeros((1, 1, 1), dtype=np.float32)
            tex = gfx.Texture(placeholder, dim=2, format="1xf4")
            self.node_2d: gfx.WorldObject | None = gfx.Image(
                gfx.Geometry(grid=tex),
                gfx.ImageBasicMaterial(
                    clim=appearance.clim,
                    map=colormap,
                    interpolation=appearance.interpolation,
                    pick_write=visual_model.pick_write,
                ),
            )
            self.node_3d: gfx.WorldObject | None = None

        else:  # render_mode == "3d"
            # Placeholder 2x2x2 texture -- replaced on first on_data_ready.
            placeholder = np.zeros((2, 2, 2), dtype=np.float32)
            tex = gfx.Texture(placeholder, dim=3, format="1xf4")
            self.node_3d = gfx.Volume(
                gfx.Geometry(grid=tex),
                gfx.VolumeMipMaterial(
                    clim=appearance.clim,
                    map=colormap,
                    interpolation=appearance.interpolation,
                    pick_write=visual_model.pick_write,
                ),
            )
            self.node_2d = None

        # Node matrix is set lazily on first build_slice_request when we
        # know the displayed axes.  Identity is fine as a placeholder.

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_levels(self) -> int:
        """Always 1 -- single-resolution in-memory store."""
        return 1

    # ------------------------------------------------------------------
    # Cancellation stubs (no brick cache to release for in-memory data)
    # ------------------------------------------------------------------

    def cancel_pending(self) -> None:
        """No-op -- in-memory visuals have no reserved GPU brick slots."""

    def cancel_pending_2d(self) -> None:
        """No-op -- in-memory visuals have no reserved GPU brick slots."""

    # ------------------------------------------------------------------
    # Node matrix update (Option C -- lazy, displayed-axes-aware)
    # ------------------------------------------------------------------

    def _update_node_matrix(self, displayed_axes: tuple[int, ...]) -> None:
        """Recompute and apply the pygfx node matrix for *displayed_axes*.

        Called from ``build_slice_request*`` when displayed axes change,
        and from ``on_transform_changed`` when the transform is updated.
        """
        self._last_displayed_axes = displayed_axes
        sub = self._transform.set_slice(displayed_axes)
        m = _pygfx_matrix(sub)
        if self.node_3d is not None:
            self.node_3d.local.matrix = m
        if self.node_2d is not None:
            self.node_2d.local.matrix = m

    # ------------------------------------------------------------------
    # Planning -- build ChunkRequests (synchronous, < 1 ms)
    # ------------------------------------------------------------------

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
        """Return a single ChunkRequest for the full 2-D slice.

        All camera/viewport parameters are accepted for interface
        compatibility but are unused.

        Parameters
        ----------
        camera_pos_world : np.ndarray
            Unused. Accepted for interface compatibility.
        viewport_width_px : float
            Unused. Accepted for interface compatibility.
        world_width : float
            Unused. Accepted for interface compatibility.
        view_min_world : np.ndarray or None
            Unused. Accepted for interface compatibility.
        view_max_world : np.ndarray or None
            Unused. Accepted for interface compatibility.
        dims_state : DimsState
            Current dimension state. Determines which axes are displayed
            and which are sliced (and at what index).
        lod_bias : float
            Unused. Accepted for interface compatibility.
        force_level : int or None
            Unused. Accepted for interface compatibility.
        use_culling : bool
            Unused. Accepted for interface compatibility.

        Returns
        -------
        list[ChunkRequest]
            Always contains exactly one element.
        """
        displayed = dims_state.selection.displayed_axes
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)

        # Transform slice indices from world space to data space.
        transformed_indices = _transform_slice_indices(
            dims_state.selection.slice_indices,
            self._transform,
            self._data_store.shape,
        )
        transformed_dims = DimsState(
            axis_labels=dims_state.axis_labels,
            selection=AxisAlignedSelectionState(
                displayed_axes=dims_state.selection.displayed_axes,
                slice_indices=transformed_indices,
            ),
        )
        axis_selections = _build_axis_selections(
            transformed_dims, self._data_store.shape
        )
        return [
            ChunkRequest(
                chunk_request_id=uuid4(),
                slice_request_id=uuid4(),
                scale_index=0,
                axis_selections=axis_selections,
            )
        ]

    def build_slice_request(
        self,
        camera_pos_world: np.ndarray,
        frustum_corners_world: np.ndarray | None,
        thresholds: list[float] | None,
        dims_state: DimsState | None = None,
        force_level: int | None = None,
    ) -> list[ChunkRequest]:
        """Return a single ChunkRequest for the full 3-D sub-volume.

        All camera/frustum parameters are accepted for interface
        compatibility but are unused.

        Parameters
        ----------
        camera_pos_world : np.ndarray
            Unused. Accepted for interface compatibility.
        frustum_corners_world : np.ndarray or None
            Unused. Accepted for interface compatibility.
        thresholds : list[float] or None
            Unused. Accepted for interface compatibility.
        dims_state : DimsState or None
            Current dimension state. If ``None`` (e.g. during headless
            tests), all axes are treated as displayed.
        force_level : int or None
            Unused. Accepted for interface compatibility.

        Returns
        -------
        list[ChunkRequest]
            Always contains exactly one element.
        """
        if dims_state is None:
            # Fallback: treat all axes as displayed, no slicing.
            ndim = self._data_store.ndim
            axis_selections = tuple(
                (0, self._data_store.shape[ax]) for ax in range(ndim)
            )
        else:
            displayed = dims_state.selection.displayed_axes
            if displayed != self._last_displayed_axes:
                self._update_node_matrix(displayed)

            # Transform slice indices from world space to data space.
            transformed_indices = _transform_slice_indices(
                dims_state.selection.slice_indices,
                self._transform,
                self._data_store.shape,
            )
            transformed_dims = DimsState(
                axis_labels=dims_state.axis_labels,
                selection=AxisAlignedSelectionState(
                    displayed_axes=dims_state.selection.displayed_axes,
                    slice_indices=transformed_indices,
                ),
            )
            axis_selections = _build_axis_selections(
                transformed_dims, self._data_store.shape
            )

        return [
            ChunkRequest(
                chunk_request_id=uuid4(),
                slice_request_id=uuid4(),
                scale_index=0,
                axis_selections=axis_selections,
            )
        ]

    # ------------------------------------------------------------------
    # Commit -- receive data from AsyncSlicer and upload to GPU
    # ------------------------------------------------------------------

    def on_data_ready(self, batch: list[tuple[ChunkRequest, np.ndarray]]) -> None:
        """Upload a 3-D array to the pygfx Volume node.

        Called on the main thread by ``SliceCoordinator`` after the
        ``AsyncSlicer`` completes the read. Replaces the node geometry
        entirely.

        Parameters
        ----------
        batch : list of (ChunkRequest, np.ndarray)
            Always contains exactly one element for this visual type.
        """
        if not batch or self.node_3d is None:
            return

        _request, data = batch[0]

        # pygfx Volume expects (W, H, D) -- transpose from numpy (D, H, W).
        data_wgpu = np.ascontiguousarray(data.T)
        tex = gfx.Texture(data_wgpu, dim=3, format="1xf4")
        self.node_3d.geometry = gfx.Geometry(grid=tex)

    def on_data_ready_2d(self, batch: list[tuple[ChunkRequest, np.ndarray]]) -> None:
        """Upload a 2-D slice to the pygfx Image node.

        Called on the main thread by ``SliceCoordinator`` after the
        ``AsyncSlicer`` completes the read.

        Parameters
        ----------
        batch : list of (ChunkRequest, np.ndarray)
            Always contains exactly one element for this visual type.
        """
        if not batch or self.node_2d is None:
            return

        _request, data = batch[0]

        # pygfx Image expects (W, H, 1) -- transpose H<->W and add channel dim.
        data_wgpu = np.ascontiguousarray(data.T[:, :, np.newaxis])
        tex = gfx.Texture(data_wgpu, dim=2, format="1xf4")
        self.node_2d.geometry = gfx.Geometry(grid=tex)

    # ------------------------------------------------------------------
    # Transform event handler
    # ------------------------------------------------------------------

    def on_transform_changed(self, event: TransformChangedEvent) -> None:
        """Update stored transform and pygfx node matrix.

        Parameters
        ----------
        event : TransformChangedEvent
            Carries the new ``AffineTransform``.
        """
        self._transform = event.transform
        if self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)

    # ------------------------------------------------------------------
    # Appearance and visibility event handlers
    # ------------------------------------------------------------------

    def on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        """Apply a pure GPU-side appearance update (no reslice needed).

        Handles ``clim``, ``color_map``, and ``interpolation``.
        Unrecognised field names are silently ignored.

        Parameters
        ----------
        event : AppearanceChangedEvent
            Carries ``field_name`` and ``new_value``.
        """
        node = self.node_2d if self._render_mode == "2d" else self.node_3d
        if node is None:
            return
        material = node.material

        if event.field_name == "clim":
            material.clim = event.new_value
        elif event.field_name == "color_map":
            material.map = _make_colormap(event.new_value)
        elif event.field_name == "interpolation":
            material.interpolation = event.new_value
        # "visible" is handled by on_visibility_changed; ignore here.

    def on_visibility_changed(self, event: VisualVisibilityChangedEvent) -> None:
        """Toggle node visibility.

        Parameters
        ----------
        event : VisualVisibilityChangedEvent
            Carries ``visible`` bool.
        """
        node = self.node_2d if self._render_mode == "2d" else self.node_3d
        if node is not None:
            node.visible = event.visible
