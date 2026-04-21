# src/cellier/v2/render/visuals/_points_memory.py
from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx

from cellier.v2.data.points._points_requests import PointsSliceRequest

if TYPE_CHECKING:
    from cellier.v2._state import DimsState
    from cellier.v2.data.points._points_requests import PointsData
    from cellier.v2.events._events import (
        AppearanceChangedEvent,
        TransformChangedEvent,
        VisualVisibilityChangedEvent,
    )
    from cellier.v2.transform import AffineTransform
    from cellier.v2.visuals._points_memory import PointsMarkerAppearance, PointsVisual

# Placeholder geometry — one invisible point so pygfx never sees an empty
# geometry buffer.
_PLACEHOLDER_POSITIONS = np.zeros((1, 3), dtype=np.float32)


def _pygfx_matrix(transform: AffineTransform) -> np.ndarray:
    """Embed a 2-D or 3-D AffineTransform into a 4x4 pygfx matrix.

    Reverses data axis order (z, y, x) → pygfx (x, y, z).
    Identical to the helper in _mesh_memory.py — keep in sync or
    extract to a shared utility.
    """
    nd = transform.ndim
    src = transform.matrix
    swap = list(reversed(range(nd)))
    m = np.eye(4, dtype=np.float32)
    for dst_i, src_i in enumerate(swap):
        for dst_j, src_j in enumerate(swap):
            m[dst_i, dst_j] = src[src_i, src_j]
        m[dst_i, 3] = src[src_i, nd]
    return m


def _build_material(appearance: PointsMarkerAppearance) -> gfx.PointsMaterial:
    """Construct a PointsMaterial from a PointsMarkerAppearance."""
    return gfx.PointsMaterial(
        size=appearance.size,
        size_space=appearance.size_space,
        color=appearance.color,
        color_mode=appearance.color_mode,
        opacity=appearance.opacity,
        depth_test=appearance.depth_test,
        depth_write=appearance.depth_write,
        depth_compare=appearance.depth_compare,
    )


class GFXPointsMemoryVisual:
    """Render-layer visual for one PointsVisual backed by in-memory points data.

    Uses a single ``gfx.Points`` node for both 2D and 3D modes.
    ``get_node_for_dims`` always returns ``self.node``; SceneManager.swap_node
    no-ops because ``old_node is new_node``.  The subsequent reslice updates
    geometry and swaps the material between ``_material`` and
    ``_empty_material``.

    Coordinate convention
    ---------------------
    The store delivers positions in data-axis order: column 0 is axis 0 (z),
    column 1 is axis 1 (y), column 2 is axis 2 (x).

    In ``_commit``:
    - **3D path** — ``positions[:, [2, 1, 0]]`` reverses to pygfx ``(x, y, z)``
      order.  This is the single axis-reversal site; do not reorder elsewhere.
    - **2D path** — incoming ``(N, 2)`` positions (already projected by the
      store) are padded with a zero column.  The node matrix handles
      orientation; no column reversal is applied.

    Parameters
    ----------
    visual_model : PointsVisual
        Associated model-layer visual.
    render_modes : set[str]
        ``{"2d"}``, ``{"3d"}``, or ``{"2d", "3d"}``.
    transform : AffineTransform
        Data-to-world transform. Must cover all data axes.
    """

    #: In-memory visuals are cheap to reslice and must never be cancelled.
    #: SliceCoordinator.submit() reads this flag before calling cancel_visual.
    cancellable: bool = False

    def __init__(
        self,
        visual_model: PointsVisual,
        render_modes: set[str],
        transform: AffineTransform,
    ) -> None:
        invalid = render_modes - {"2d", "3d"}
        if invalid or not render_modes:
            raise ValueError(
                f"render_modes must be a non-empty subset of {{'2d','3d'}}, "
                f"got {render_modes!r}"
            )

        self.visual_model_id: UUID = visual_model.id
        self.render_modes: set[str] = render_modes
        self._transform: AffineTransform = transform
        self._last_displayed_axes: tuple[int, ...] | None = None

        appearance = visual_model.appearance
        self._material = _build_material(appearance)
        self._empty_material = gfx.PointsMaterial(color=(0, 0, 0, 0), opacity=0.0)

        # Track color_mode and size_mode as instance state to detect transitions.
        self._current_color_mode: str = appearance.color_mode
        self._current_size_mode: str = "uniform"
        self._is_empty: bool = True

        geom = gfx.Geometry(positions=_PLACEHOLDER_POSITIONS.copy())
        self.node = gfx.Points(geom, self._empty_material)
        self.node.render_order = appearance.render_order

        # Both attributes point to the same node.
        # SceneManager.swap_node's old_node is new_node guard makes dim-toggling
        # a no-op for this visual.
        self.node_2d = self.node
        self.node_3d = self.node

    # ------------------------------------------------------------------
    # LOD
    # ------------------------------------------------------------------

    @property
    def n_levels(self) -> int:
        """Always 1 — single-resolution in-memory store."""
        return 1

    # ------------------------------------------------------------------
    # Node selection
    # ------------------------------------------------------------------

    def get_node_for_dims(self, displayed_axes: tuple[int, ...]) -> gfx.Points:
        """Return the node for the given displayed axes — always self.node.

        SceneManager.swap_node will no-op because the returned node is
        the same object as the currently active node.  The reslice that
        follows updates the geometry and material.

        Parameters
        ----------
        displayed_axes : tuple[int, ...]
            New displayed axes.

        Returns
        -------
        gfx.Points
            Always self.node.
        """
        if displayed_axes != self._last_displayed_axes:
            self._update_node_matrix(displayed_axes)
        return self.node

    # ------------------------------------------------------------------
    # Node matrix
    # ------------------------------------------------------------------

    def _update_node_matrix(self, displayed_axes: tuple[int, ...]) -> None:
        self._last_displayed_axes = displayed_axes
        sub = self._transform.set_slice(displayed_axes)
        self.node.local.matrix = _pygfx_matrix(sub)

    # ------------------------------------------------------------------
    # Slice request building
    # ------------------------------------------------------------------

    def _build_request(self, dims_state: DimsState) -> PointsSliceRequest:
        shared_id = uuid4()
        return PointsSliceRequest(
            slice_request_id=shared_id,
            chunk_request_id=shared_id,
            scale_index=0,
            displayed_axes=dims_state.selection.displayed_axes,
            slice_indices=dict(dims_state.selection.slice_indices),
            thickness=0.5,
        )

    def build_slice_request(
        self,
        camera_pos_world: np.ndarray,
        frustum_corners_world: np.ndarray | None,
        thresholds: list[float] | None,
        dims_state: DimsState,
        force_level: int | None = None,
    ) -> list[PointsSliceRequest]:
        """3-D planning path — returns one PointsSliceRequest."""
        displayed = dims_state.selection.displayed_axes
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)
        return [self._build_request(dims_state)]

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
    ) -> list[PointsSliceRequest]:
        """2-D planning path — returns one PointsSliceRequest."""
        displayed = dims_state.selection.displayed_axes
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)
        return [self._build_request(dims_state)]

    # ------------------------------------------------------------------
    # Commit — shared logic for both callbacks
    # ------------------------------------------------------------------

    def _commit(self, points_data: PointsData, is_2d: bool) -> None:
        """Upload PointsData to self.node.

        Pads 2D positions to 3D with z=0.
        Applies axis reversal for 3D data to match pygfx coordinate order.
        Swaps material between _material and _empty_material as needed.
        Updates _current_color_mode when the incoming data changes mode.

        Coordinate convention (DO NOT reorder positions elsewhere):
        - 3D path: positions[:, [2, 1, 0]] reverses (z, y, x) → (x, y, z)
          for pygfx.
        - 2D path: zero-pad to 3D; swap columns [1, 0, 2].
        """
        positions = points_data.positions
        n_points = positions.shape[0]
        n_dims = positions.shape[1]

        if n_dims == 2:
            # 2D path — pad displayed-plane coords with z=0.
            # Node matrix handles axis orientation; no reversal needed.
            zeros = np.zeros((n_points, 1), dtype=np.float32)
            pos3d = np.concatenate([positions, zeros], axis=1)
            pos3d = pos3d[:, [1, 0, 2]]
        else:
            # 3D path — reverse (z, y, x) data order to (x, y, z) pygfx order.
            # This is the single axis-reversal site for points. Do not reorder
            # positions anywhere else in the pipeline.
            pos3d = np.ascontiguousarray(positions)[:, [2, 1, 0]]

        geom_kwargs: dict = {"positions": pos3d}

        colors = points_data.colors
        if colors is not None:
            geom_kwargs["colors"] = np.ascontiguousarray(colors)

        sizes = points_data.sizes
        if sizes is not None:
            geom_kwargs["sizes"] = np.ascontiguousarray(sizes)

        self.node.geometry = gfx.Geometry(**geom_kwargs)

        # Update color_mode on live material if it changed.
        incoming_color_mode = (
            points_data.color_mode if colors is not None else "uniform"
        )
        if incoming_color_mode != self._current_color_mode and not points_data.is_empty:
            self._current_color_mode = incoming_color_mode
            self._material.color_mode = incoming_color_mode

        # Select material.
        target = self._empty_material if points_data.is_empty else self._material
        if self.node.material is not target:
            self.node.material = target

        self._is_empty = points_data.is_empty

    def on_data_ready(self, batch: list[tuple[PointsSliceRequest, PointsData]]) -> None:
        """3-D callback — called on the main thread by SliceCoordinator."""
        if not batch:
            return
        _, data = batch[0]
        self._commit(data, is_2d=False)

    def on_data_ready_2d(
        self, batch: list[tuple[PointsSliceRequest, PointsData]]
    ) -> None:
        """2-D callback — called on the main thread by SliceCoordinator."""
        if not batch:
            return
        _, data = batch[0]
        self._commit(data, is_2d=True)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_transform_changed(self, event: TransformChangedEvent) -> None:
        self._transform = event.transform
        if self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)

    def on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        """Apply appearance field changes to the live material.

        ``color_mode`` changes also update ``_current_color_mode`` so
        the next ``_commit`` call does not overwrite them.

        Note: ``size_space`` is a constructor-only parameter on
        ``gfx.PointsMaterial``; changes to ``size_space`` require
        rebuilding the material.
        """
        name = event.field_name
        val = event.new_value
        if name == "color":
            self._material.color = val
        elif name == "color_mode":
            self._material.color_mode = val
            self._current_color_mode = val
        elif name == "opacity":
            self._material.opacity = val
        elif name == "size":
            self._material.size = val
        elif name == "depth_test":
            self._material.depth_test = val
        elif name == "depth_write":
            self._material.depth_write = val
        elif name == "depth_compare":
            self._material.depth_compare = val
        elif name == "render_order":
            self.node.render_order = val

    def on_visibility_changed(self, event: VisualVisibilityChangedEvent) -> None:
        self.node.visible = event.visible

    # ------------------------------------------------------------------
    # No-op cancellation stubs (no brick cache)
    # ------------------------------------------------------------------

    def cancel_pending(self) -> None:
        """No-op — in-memory visuals have no reserved GPU brick slots."""

    def cancel_pending_2d(self) -> None:
        """No-op — in-memory visuals have no reserved GPU brick slots."""
