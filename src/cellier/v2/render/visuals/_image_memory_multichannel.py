# src/cellier/v2/render/visuals/_image_memory_multichannel.py
"""Render-layer visual for multichannel in-memory images."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx

from cellier.v2._state import AxisAlignedSelectionState, DimsState
from cellier.v2.data.image._image_requests import ChunkRequest
from cellier.v2.render.visuals._image_memory import (
    _pygfx_matrix,
    _transform_slice_indices,
)
from cellier.v2.render.visuals._multichannel_utils import (
    apply_channel_appearance_2d,
    apply_channel_appearance_3d,
    build_axis_selections_for_channel,
    channel_index_from_request,
    make_channel_group_2d,
    make_channel_group_3d,
)

if TYPE_CHECKING:
    from cellier.v2.data.image._image_memory_store import ImageMemoryStore
    from cellier.v2.events._events import (
        AABBChangedEvent,
        TransformChangedEvent,
        VisualVisibilityChangedEvent,
    )
    from cellier.v2.transform import AffineTransform
    from cellier.v2.visuals._channel_appearance import ChannelAppearance
    from cellier.v2.visuals._image_memory import MultichannelImageVisual


class GFXMultichannelImageMemoryVisual:
    """Render-layer visual for a ``MultichannelImageVisual`` backed by memory.

    Maintains a fixed-size pool of pygfx nodes (one per channel slot).
    Channels are mapped to slots at construction time and when the channel dict
    is replaced.

    Parameters
    ----------
    visual_model : MultichannelImageVisual
        Associated model-layer visual.
    data_store : ImageMemoryStore
        The backing data store.
    render_modes : set[str]
        Which nodes to build: ``{"2d"}``, ``{"3d"}``, or ``{"2d", "3d"}``.
    transform : AffineTransform or None
        Data-to-world transform.
    """

    cancellable: bool = True

    def __init__(
        self,
        visual_model: MultichannelImageVisual,
        data_store: ImageMemoryStore,
        render_modes: set[str],
        transform: AffineTransform | None = None,
    ) -> None:
        invalid = render_modes - {"2d", "3d"}
        if invalid or not render_modes:
            raise ValueError(
                f"render_modes must be a non-empty subset of {{'2d', '3d'}}, "
                f"got {render_modes!r}"
            )

        self.visual_model_id: UUID = visual_model.id
        self.render_modes = render_modes
        self._data_store = data_store
        self._visual_model = visual_model
        self._channel_axis = visual_model.channel_axis

        if transform is None:
            from cellier.v2.transform import AffineTransform as _AT

            transform = _AT.identity(ndim=data_store.ndim)
        elif transform.ndim < data_store.ndim:
            transform = transform.expand_dims(data_store.ndim)
        self._transform: AffineTransform = transform

        self._last_displayed_axes: tuple[int, ...] | None = None
        self._current_slice_request_id_3d: UUID | None = None
        self._current_slice_request_id_2d: UUID | None = None

        # Pool state
        self._pool_2d: list[gfx.Image] = []
        self._pool_3d: list[gfx.Volume] = []
        self._group_2d: gfx.Group | None = None
        self._group_3d: gfx.Group | None = None
        self._channel_to_slot_2d: dict[int, int] = {}
        self._channel_to_slot_3d: dict[int, int] = {}
        self._free_slots_2d: list[int] = []
        self._free_slots_3d: list[int] = []
        # channel_index → psygnal handler
        self._appearance_callbacks: dict[int, object] = {}

        max_2d = visual_model.max_channels_2d
        max_3d = visual_model.max_channels_3d

        if "2d" in render_modes:
            self._group_2d, self._pool_2d = make_channel_group_2d(
                visual_model.channels, max_2d
            )
            self._free_slots_2d = list(range(max_2d))

        if "3d" in render_modes:
            self._group_3d, self._pool_3d = make_channel_group_3d(
                visual_model.channels, max_3d
            )
            self._free_slots_3d = list(range(max_3d))

        # Claim initial channels
        for ch_idx, appearance in visual_model.channels.items():
            self._claim_channel(ch_idx, appearance)

        # Subscribe to whole-dict replacement
        visual_model.events.channels.connect(self._on_channels_replaced)

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def _claim_channel(self, ch_idx: int, appearance: ChannelAppearance) -> None:
        if "2d" in self.render_modes and self._free_slots_2d:
            slot = self._free_slots_2d.pop(0)
            self._channel_to_slot_2d[ch_idx] = slot
            apply_channel_appearance_2d(self._pool_2d[slot], appearance)
            self._pool_2d[slot].visible = appearance.visible

        if "3d" in self.render_modes and self._free_slots_3d:
            slot = self._free_slots_3d.pop(0)
            self._channel_to_slot_3d[ch_idx] = slot
            apply_channel_appearance_3d(self._pool_3d[slot], appearance)
            self._pool_3d[slot].visible = appearance.visible

        self._subscribe_to_channel_appearance(ch_idx, appearance)

    def _release_channel(self, ch_idx: int) -> None:
        if ch_idx in self._channel_to_slot_2d:
            slot = self._channel_to_slot_2d.pop(ch_idx)
            self._pool_2d[slot].visible = False
            self._free_slots_2d.append(slot)

        if ch_idx in self._channel_to_slot_3d:
            slot = self._channel_to_slot_3d.pop(ch_idx)
            self._pool_3d[slot].visible = False
            self._free_slots_3d.append(slot)

        handler = self._appearance_callbacks.pop(ch_idx, None)
        if handler is not None:
            appearance = self._visual_model.channels.get(ch_idx)
            if appearance is not None:
                try:
                    appearance.events.disconnect(handler)
                except Exception:
                    pass

    def _subscribe_to_channel_appearance(
        self, ch_idx: int, appearance: ChannelAppearance
    ) -> None:
        def _handler(info, idx: int = ch_idx) -> None:
            self._on_channel_appearance_changed(idx, info)

        appearance.events.connect(_handler)
        self._appearance_callbacks[ch_idx] = _handler

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_channels_replaced(self, new_channels: dict) -> None:
        old = set(self._channel_to_slot_2d) | set(self._channel_to_slot_3d)
        new = set(new_channels)
        for idx in old - new:
            self._release_channel(idx)
        for idx in new - old:
            self._claim_channel(idx, new_channels[idx])

    def _on_channel_appearance_changed(self, ch_idx: int, info) -> None:
        field_name: str = info.signal.name
        new_value = info.args[0]

        if "2d" in self.render_modes and ch_idx in self._channel_to_slot_2d:
            node_2d = self._pool_2d[self._channel_to_slot_2d[ch_idx]]
            if field_name == "clim":
                node_2d.material.clim = new_value
            elif field_name == "colormap":
                from cellier.v2.render.visuals._image_memory import _make_colormap

                node_2d.material.map = _make_colormap(new_value)
            elif field_name == "opacity":
                node_2d.material.opacity = new_value
            elif field_name == "transparency_mode":
                node_2d.material.alpha_mode = new_value
            elif field_name == "visible":
                node_2d.visible = new_value

        if "3d" in self.render_modes and ch_idx in self._channel_to_slot_3d:
            node_3d = self._pool_3d[self._channel_to_slot_3d[ch_idx]]
            if field_name == "clim":
                node_3d.material.clim = new_value
            elif field_name == "colormap":
                from cellier.v2.render.visuals._image_memory import _make_colormap

                node_3d.material.map = _make_colormap(new_value)
            elif field_name == "opacity":
                node_3d.material.opacity = new_value
            elif field_name == "transparency_mode":
                node_3d.material.alpha_mode = new_value
            elif field_name == "visible":
                node_3d.visible = new_value

    # ------------------------------------------------------------------
    # Cancellation stubs
    # ------------------------------------------------------------------

    def cancel_pending(self) -> None:
        """No-op — in-memory visuals have no reserved GPU brick slots."""

    def cancel_pending_2d(self) -> None:
        """No-op — in-memory visuals have no reserved GPU brick slots."""

    # ------------------------------------------------------------------
    # Node matrix
    # ------------------------------------------------------------------

    def _update_node_matrix(self, displayed_axes: tuple[int, ...]) -> None:
        self._last_displayed_axes = displayed_axes
        sub = self._transform.select_axes(displayed_axes)
        m = _pygfx_matrix(sub)
        if self._group_3d is not None:
            self._group_3d.local.matrix = m
        if self._group_2d is not None:
            self._group_2d.local.matrix = m

    # ------------------------------------------------------------------
    # Node selection
    # ------------------------------------------------------------------

    def get_node_for_dims(self, displayed_axes: tuple[int, ...]) -> gfx.Group | None:
        """Return the pre-built Group node appropriate for ``displayed_axes``.

        Parameters
        ----------
        displayed_axes : tuple[int, ...]
            The new set of displayed axes.

        Returns
        -------
        gfx.Group or None
            ``_group_3d`` when ``len(displayed_axes) == 3``, else ``_group_2d``.
        """
        if len(displayed_axes) == 3:
            node = self._group_3d
        else:
            node = self._group_2d

        if displayed_axes != self._last_displayed_axes:
            self._update_node_matrix(displayed_axes)

        return node

    def get_channel_node_for_dims(
        self, channel_index: int, displayed_axes: tuple[int, ...]
    ) -> gfx.Image | gfx.Volume | None:
        """Return the individual node for one channel.

        Parameters
        ----------
        channel_index : int
            Channel index to look up.
        displayed_axes : tuple[int, ...]
            Determines whether to return a 2D or 3D node.

        Returns
        -------
        gfx.Image or gfx.Volume or None
        """
        if len(displayed_axes) == 3:
            slot = self._channel_to_slot_3d.get(channel_index)
            if slot is None:
                return None
            return self._pool_3d[slot]
        slot = self._channel_to_slot_2d.get(channel_index)
        if slot is None:
            return None
        return self._pool_2d[slot]

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _make_channel_requests(
        self, dims_state: DimsState, slice_request_id
    ) -> list[ChunkRequest]:
        transformed_indices = _transform_slice_indices(
            dims_state.selection.slice_indices,
            self._transform,
            self._data_store.shape,
        )
        transformed_state = DimsState(
            axis_labels=dims_state.axis_labels,
            selection=AxisAlignedSelectionState(
                displayed_axes=dims_state.selection.displayed_axes,
                slice_indices=transformed_indices,
            ),
        )
        requests: list[ChunkRequest] = []
        for ch_idx, appearance in self._visual_model.channels.items():
            if not appearance.visible:
                continue
            axis_selections = build_axis_selections_for_channel(
                transformed_state,
                self._data_store.shape,
                self._channel_axis,
                ch_idx,
            )
            requests.append(
                ChunkRequest(
                    chunk_request_id=uuid4(),
                    slice_request_id=slice_request_id,
                    scale_index=0,
                    axis_selections=axis_selections,
                )
            )
        return requests

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
        """Return one ChunkRequest per visible channel for the 2D slice."""
        displayed = dims_state.selection.displayed_axes
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)
        sid = uuid4()
        self._current_slice_request_id_2d = sid
        return self._make_channel_requests(dims_state, sid)

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
        """Return one ChunkRequest per visible channel for the 3D sub-volume."""
        if dims_state is None:
            ndim = self._data_store.ndim
            axis_selections_base = tuple(
                (0, self._data_store.shape[ax]) if ax != self._channel_axis else 0
                for ax in range(ndim)
            )
            sid = uuid4()
            self._current_slice_request_id_3d = sid
            requests = []
            for ch_idx in self._visual_model.channels:
                sels = tuple(
                    ch_idx if ax == self._channel_axis else v
                    for ax, v in enumerate(axis_selections_base)
                )
                requests.append(
                    ChunkRequest(
                        chunk_request_id=uuid4(),
                        slice_request_id=sid,
                        scale_index=0,
                        axis_selections=sels,
                    )
                )
            return requests

        displayed = dims_state.selection.displayed_axes
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)
        sid = uuid4()
        self._current_slice_request_id_3d = sid
        return self._make_channel_requests(dims_state, sid)

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def on_data_ready_2d(self, batch: list[tuple[ChunkRequest, np.ndarray]]) -> None:
        """Upload 2D slices to the correct pool nodes."""
        for request, data in batch:
            if request.slice_request_id != self._current_slice_request_id_2d:
                continue
            ch_idx = channel_index_from_request(request, self._channel_axis)
            slot = self._channel_to_slot_2d.get(ch_idx)
            if slot is None:
                continue
            node = self._pool_2d[slot]
            data_wgpu = np.ascontiguousarray(data[:, :, np.newaxis])
            tex = gfx.Texture(data_wgpu, dim=2, format="1xf4")
            node.geometry = gfx.Geometry(grid=tex)

    def on_data_ready(self, batch: list[tuple[ChunkRequest, np.ndarray]]) -> None:
        """Upload 3D volumes to the correct pool nodes."""
        for request, data in batch:
            if request.slice_request_id != self._current_slice_request_id_3d:
                continue
            ch_idx = channel_index_from_request(request, self._channel_axis)
            slot = self._channel_to_slot_3d.get(ch_idx)
            if slot is None:
                continue
            node = self._pool_3d[slot]
            data_wgpu = np.ascontiguousarray(data.T)
            tex = gfx.Texture(data_wgpu, dim=3, format="1xf4")
            node.geometry = gfx.Geometry(grid=tex)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_transform_changed(self, event: TransformChangedEvent) -> None:
        """Update stored transform and pygfx node matrix."""
        self._transform = event.transform
        if self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)

    def on_visibility_changed(self, event: VisualVisibilityChangedEvent) -> None:
        """Toggle Group node visibility."""
        for node in (self._group_2d, self._group_3d):
            if node is not None:
                node.visible = event.visible

    def on_aabb_changed(self, event: AABBChangedEvent) -> None:
        """No-op — multichannel memory visual has no AABB wireframe."""

    def tick(self) -> None:
        """No per-frame state to advance."""
