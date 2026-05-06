# src/cellier/v2/render/visuals/_image_multiscale_multichannel.py
"""Render-layer visual for multichannel multiscale images."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pygfx as gfx

from cellier.v2.render.visuals._image import GFXMultiscaleImageVisual
from cellier.v2.render.visuals._image_memory import _make_colormap
from cellier.v2.transform._axis_order import select_axes

if TYPE_CHECKING:
    import numpy as np

    from cellier.v2._state import DimsState
    from cellier.v2.data.image._image_requests import ChunkRequest
    from cellier.v2.events._events import (
        AABBChangedEvent,
        TransformChangedEvent,
        VisualVisibilityChangedEvent,
    )
    from cellier.v2.transform import AffineTransform
    from cellier.v2.visuals._channel_appearance import ChannelAppearance
    from cellier.v2.visuals._image import MultichannelMultiscaleImageVisual


class GFXMultichannelMultiscaleImageVisual:
    """Render-layer visual for one ``MultichannelMultiscaleImageVisual``.

    Maintains a fixed-size pool of ``GFXMultiscaleImageVisual`` slots, one per
    active channel. A single "planner" slot (always slot 0) runs LOD selection
    via ``_plan_bricks`` / ``_plan_tiles_2d``; every occupied slot participates
    in data delivery via its own ``_materialize_brick_requests`` /
    ``_materialize_tile_requests``.

    Parameters
    ----------
    visual_model : MultichannelMultiscaleImageVisual
        Associated model-layer visual.
    level_shapes : list[tuple[int, ...]]
        Full nD shape per level, finest first (includes channel axis).
    render_modes : set[str]
        Which nodes to build: ``{"2d"}``, ``{"3d"}``, or ``{"2d", "3d"}``.
    displayed_axes : tuple[int, ...]
        Initially displayed data axes.
    transform : AffineTransform or None
        Spatial (non-channel) data-to-world transform.  Will be expanded to
        match ``len(level_shapes[0])`` dimensions.

    Notes
    -----
    This implementation assumes the channel axis is a *leading* axis (e.g.
    CZYX, channel_axis=0).  ``AffineTransform.expand_dims`` inserts an
    identity leading axis to expand spatial transforms to full data ndim; this
    is only correct when all non-spatial axes come before the spatial ones.
    """

    cancellable: bool = True

    def __init__(
        self,
        visual_model: MultichannelMultiscaleImageVisual,
        level_shapes: list[tuple[int, ...]],
        render_modes: set[str],
        displayed_axes: tuple[int, ...],
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
        self._visual_model = visual_model
        self._channel_axis = visual_model.channel_axis
        self._full_level_shapes = list(level_shapes)

        # Expand spatial level_transforms to full data ndim so that
        # _build_world_to_level_transforms inside each slot composes correctly.
        full_ndim = len(level_shapes[0])
        self._expanded_level_transforms: list[AffineTransform] = [
            lt.expand_dims(full_ndim) for lt in visual_model.level_transforms
        ]

        # Spatial transform: let each slot expand it to full_ndim as needed.
        self._transform = transform  # may be None; slots handle expansion

        # Pool state
        rc = visual_model.render_config
        max_slots = max(
            visual_model.max_channels_2d if "2d" in render_modes else 0,
            visual_model.max_channels_3d if "3d" in render_modes else 0,
        )
        per_slot_3d = max(1, rc.gpu_budget_bytes // max_slots)
        per_slot_2d = max(1, rc.gpu_budget_bytes_2d // max_slots)

        # Build shared geometry objects (same for all slots).
        self._slots: list[GFXMultiscaleImageVisual] = []
        for _ in range(max_slots):
            slot = self._make_slot(
                visual_model=visual_model,
                level_shapes=level_shapes,
                render_modes=render_modes,
                displayed_axes=displayed_axes,
                transform=transform,
                gpu_budget_bytes_3d=per_slot_3d,
                gpu_budget_bytes_2d=per_slot_2d,
            )
            self._slots.append(slot)

        self._planner_slot_index: int = 0
        self._channel_to_slot: dict[int, int] = {}
        self._free_slots: list[int] = list(range(max_slots))
        self._appearance_callbacks: dict[int, object] = {}

        self._last_displayed_axes: tuple[int, ...] = displayed_axes
        self._current_slice_request_id_3d: UUID | None = None
        self._current_slice_request_id_2d: UUID | None = None

        # Build wrapper groups containing all slot nodes.
        self._group_3d: gfx.Group | None = None
        self._group_2d: gfx.Group | None = None

        if "3d" in render_modes:
            self._group_3d = gfx.Group()
            for slot in self._slots:
                if slot.node_3d is not None:
                    slot.node_3d.visible = False
                    self._group_3d.add(slot.node_3d)

        if "2d" in render_modes:
            self._group_2d = gfx.Group()
            for slot in self._slots:
                if slot.node_2d is not None:
                    slot.node_2d.visible = False
                    self._group_2d.add(slot.node_2d)

        # Claim initial channels.
        for ch_idx, appearance in visual_model.channels.items():
            self._claim_channel(ch_idx, appearance)

        # Subscribe to whole-dict channel replacement.
        visual_model.events.channels.connect(self._on_channels_replaced)

    # ------------------------------------------------------------------
    # Slot factory
    # ------------------------------------------------------------------

    def _make_slot(
        self,
        visual_model: MultichannelMultiscaleImageVisual,
        level_shapes: list[tuple[int, ...]],
        render_modes: set[str],
        displayed_axes: tuple[int, ...],
        transform: AffineTransform | None,
        gpu_budget_bytes_3d: int,
        gpu_budget_bytes_2d: int,
    ) -> GFXMultiscaleImageVisual:
        """Create one ``GFXMultiscaleImageVisual`` channel slot."""
        from cellier.v2.render.visuals._image import (
            ImageGeometry2D,
            VolumeGeometry,
        )

        rc = visual_model.render_config
        channel_axis = visual_model.channel_axis

        # Spatial displayed axes: all displayed_axes that are NOT the channel
        # axis (channel is never a displayed spatial axis).
        spatial_displayed = tuple(ax for ax in displayed_axes if ax != channel_axis)

        # Build geometry objects over the spatial displayed axes only.
        volume_geometry: VolumeGeometry | None = None
        image_geometry_2d: ImageGeometry2D | None = None

        if "3d" in render_modes and len(spatial_displayed) >= 3:
            axes_3d = spatial_displayed[-3:]
            shapes_3d = [select_axes(s, axes_3d) for s in level_shapes]
            # Use original (spatial) level_transforms for geometry; they share
            # the same per-level scale information.
            transforms_3d = [
                lt.select_axes(axes_3d) for lt in visual_model.level_transforms
            ]
            volume_geometry = VolumeGeometry(
                level_shapes=shapes_3d,
                level_transforms=transforms_3d,
                block_size=rc.block_size,
            )

        if "2d" in render_modes:
            axes_2d = spatial_displayed[-2:]
            shapes_2d = [select_axes(s, axes_2d) for s in level_shapes]
            transforms_2d = [
                lt.select_axes(axes_2d) for lt in visual_model.level_transforms
            ]
            image_geometry_2d = ImageGeometry2D(
                level_shapes=shapes_2d,
                block_size=rc.block_size,
                n_levels=len(level_shapes),
                level_transforms=transforms_2d,
            )

        return GFXMultiscaleImageVisual(
            visual_model_id=visual_model.id,
            volume_geometry=volume_geometry,
            image_geometry_2d=image_geometry_2d,
            render_modes=render_modes,
            displayed_axes=displayed_axes,
            colormap=gfx.cm.viridis,
            clim=(0.0, 1.0),
            threshold=0.5,
            interpolation=rc.interpolation,
            gpu_budget_bytes_3d=gpu_budget_bytes_3d,
            gpu_budget_bytes_2d=gpu_budget_bytes_2d,
            transform=transform,
            full_level_transforms=self._expanded_level_transforms,
            full_level_shapes=level_shapes,
            aabb_enabled=visual_model.aabb.enabled,
            aabb_color=visual_model.aabb.color,
            aabb_line_width=visual_model.aabb.line_width,
        )

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def _claim_channel(self, ch_idx: int, appearance: ChannelAppearance) -> None:
        if not self._free_slots:
            return
        slot_idx = self._free_slots.pop(0)
        self._channel_to_slot[ch_idx] = slot_idx
        slot = self._slots[slot_idx]

        # Apply appearance to the slot's material.
        cm = _make_colormap(appearance.colormap)
        if slot.material_3d is not None:
            slot.material_3d.map = cm
            slot.material_3d.clim = appearance.clim
            slot.material_3d.opacity = appearance.opacity
            slot.material_3d.alpha_mode = appearance.transparency_mode
        if slot.material_2d is not None:
            slot.material_2d.map = cm
            slot.material_2d.clim = appearance.clim
            slot.material_2d.opacity = appearance.opacity
            slot.material_2d.alpha_mode = appearance.transparency_mode
            # Multiple channel planes occupy the same world Z=0. Disable
            # both depth testing and depth writing so every plane renders
            # regardless of depth buffer state and none occludes the next
            # before blending can composite them.
            slot.material_2d.depth_test = False
            slot.material_2d.depth_write = False

        if slot.material_3d is not None:
            slot.material_3d.render_mode = appearance.render_mode_3d

        if slot.node_3d is not None:
            slot.node_3d.visible = appearance.visible
        if slot.node_2d is not None:
            slot.node_2d.visible = appearance.visible

        self._subscribe_to_channel_appearance(ch_idx, appearance)

    def _release_channel(self, ch_idx: int) -> None:
        slot_idx = self._channel_to_slot.pop(ch_idx, None)
        if slot_idx is None:
            return
        slot = self._slots[slot_idx]
        if slot.node_3d is not None:
            slot.node_3d.visible = False
        if slot.node_2d is not None:
            slot.node_2d.visible = False
        self._free_slots.append(slot_idx)

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
        old = set(self._channel_to_slot)
        new = set(new_channels)
        for idx in old - new:
            self._release_channel(idx)
        for idx in new - old:
            self._claim_channel(idx, new_channels[idx])

    def _on_channel_appearance_changed(self, ch_idx: int, info) -> None:
        field_name: str = info.signal.name
        new_value = info.args[0]
        slot_idx = self._channel_to_slot.get(ch_idx)
        if slot_idx is None:
            return
        slot = self._slots[slot_idx]

        if field_name == "clim":
            if slot.material_3d is not None:
                slot.material_3d.clim = new_value
            if slot.material_2d is not None:
                slot.material_2d.clim = new_value
        elif field_name == "colormap":
            cm = _make_colormap(new_value)
            if slot.material_3d is not None:
                slot.material_3d.map = cm
            if slot.material_2d is not None:
                slot.material_2d.map = cm
        elif field_name == "opacity":
            if slot.material_3d is not None:
                slot.material_3d.opacity = new_value
            if slot.material_2d is not None:
                slot.material_2d.opacity = new_value
        elif field_name == "render_mode_3d":
            if slot.material_3d is not None:
                slot.material_3d.render_mode = new_value
        elif field_name == "transparency_mode":
            if slot.material_3d is not None:
                slot.material_3d.alpha_mode = new_value
            if slot.material_2d is not None:
                slot.material_2d.alpha_mode = new_value
        elif field_name == "visible":
            if slot.node_3d is not None:
                slot.node_3d.visible = new_value
            if slot.node_2d is not None:
                slot.node_2d.visible = new_value

    # ------------------------------------------------------------------
    # Cancellation stubs
    # ------------------------------------------------------------------

    def cancel_pending(self) -> None:
        """Release all in-flight 3D slots across all channel slots."""
        for slot in self._slots:
            slot.cancel_pending()

    def cancel_pending_2d(self) -> None:
        """Release all in-flight 2D slots across all channel slots."""
        for slot in self._slots:
            slot.cancel_pending_2d()

    # ------------------------------------------------------------------
    # Node selection
    # ------------------------------------------------------------------

    def get_node_for_dims(self, displayed_axes: tuple[int, ...]) -> gfx.Group | None:
        """Return the wrapper Group node for *displayed_axes*.

        When displayed_axes change, rebuilds geometry on all slots and
        updates the wrapper groups if slot nodes changed.

        Parameters
        ----------
        displayed_axes : tuple[int, ...]
            len==3 → ``_group_3d``; len==2 → ``_group_2d``.
        """
        if displayed_axes != self._last_displayed_axes:
            self._rebuild_slot_geometries(displayed_axes)
        return self._group_3d if len(displayed_axes) == 3 else self._group_2d

    def _rebuild_slot_geometries(self, displayed_axes: tuple[int, ...]) -> None:
        self._last_displayed_axes = displayed_axes
        is_3d = len(displayed_axes) == 3
        group = self._group_3d if is_3d else self._group_2d
        if group is None:
            return
        for slot in self._slots:
            old_node, new_node = slot.rebuild_geometry(
                self._full_level_shapes, displayed_axes
            )
            if old_node is not new_node:
                if old_node is not None:
                    group.remove(old_node)
                if new_node is not None:
                    group.add(new_node)

    def get_channel_node_for_dims(
        self, channel_index: int, displayed_axes: tuple[int, ...]
    ) -> gfx.WorldObject | None:
        """Return the individual node for one channel.

        Parameters
        ----------
        channel_index : int
            Channel index to look up.
        displayed_axes : tuple[int, ...]
            Determines whether to return a 2D or 3D node.
        """
        slot_idx = self._channel_to_slot.get(channel_index)
        if slot_idx is None:
            return None
        slot = self._slots[slot_idx]
        return slot.node_3d if len(displayed_axes) == 3 else slot.node_2d

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

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
        """Plan 3D bricks once, then materialize for each visible channel."""
        visible_channels = {
            ch_idx: ap
            for ch_idx, ap in self._visual_model.channels.items()
            if ap.visible and ch_idx in self._channel_to_slot
        }
        if not visible_channels:
            return []

        planner = self._slots[self._planner_slot_index]

        # Ensure planner has current displayed axes.
        if dims_state is not None:
            displayed = dims_state.selection.displayed_axes
            if displayed != self._last_displayed_axes:
                self._rebuild_slot_geometries(displayed)

        brick_arr = planner._plan_bricks(
            camera_pos_world,
            frustum_corners_world,
            fov_y_rad,
            screen_height_px,
            lod_bias,
            force_level,
        )
        if not len(brick_arr):
            return []

        sid = uuid4()
        self._current_slice_request_id_3d = sid
        from itertools import zip_longest

        channel_request_lists: list[list[ChunkRequest]] = []
        for ch_idx in visible_channels:
            slot_idx = self._channel_to_slot[ch_idx]
            channel_slot = self._slots[slot_idx]
            channel_request_lists.append(
                channel_slot._materialize_brick_requests(
                    brick_arr,
                    sid,
                    dims_state,
                    fill={self._channel_axis: ch_idx},
                )
            )
        return [
            req
            for brick_group in zip_longest(*channel_request_lists)
            for req in brick_group
            if req is not None
        ]

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
        """Plan 2D tiles once, then materialize for each visible channel."""
        visible_channels = {
            ch_idx: ap
            for ch_idx, ap in self._visual_model.channels.items()
            if ap.visible and ch_idx in self._channel_to_slot
        }
        if not visible_channels:
            return []

        planner = self._slots[self._planner_slot_index]

        # Ensure planner has current displayed axes.
        displayed = dims_state.selection.displayed_axes
        if displayed != self._last_displayed_axes:
            self._rebuild_slot_geometries(displayed)

        # Set current slice coord on all slots (needed by _materialize_tile_requests).
        current_slice_coord = tuple(sorted(dims_state.selection.slice_indices.items()))
        for slot in self._slots:
            slot._current_slice_coord = current_slice_coord

        planner._frame_number += 1

        required, target_level = planner._plan_tiles_2d(
            camera_pos_world=camera_pos_world,
            viewport_width_px=viewport_width_px,
            world_width=world_width,
            view_min_world=view_min_world,
            view_max_world=view_max_world,
            lod_bias=lod_bias,
            force_level=force_level,
            use_culling=use_culling,
        )
        if not required:
            return []

        sid = uuid4()
        self._current_slice_request_id_2d = sid
        from itertools import zip_longest

        channel_request_lists: list[list[ChunkRequest]] = []
        for ch_idx in visible_channels:
            slot_idx = self._channel_to_slot[ch_idx]
            channel_slot = self._slots[slot_idx]
            channel_request_lists.append(
                channel_slot._materialize_tile_requests(
                    required,
                    target_level,
                    dims_state,
                    sid,
                    fill={self._channel_axis: ch_idx},
                )
            )
        # DEBUG print #1
        print(
            f"[MULTI-BUILD-2D] sid={str(sid)[:8]} "
            f"planner_frame={planner._frame_number} "
            f"ch_request_counts={[len(r) for r in channel_request_lists]} "
            f"total={sum(len(r) for r in channel_request_lists)}"
        )
        return [
            req
            for tile_group in zip_longest(*channel_request_lists)
            for req in tile_group
            if req is not None
        ]

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def on_data_ready(self, batch: list[tuple[ChunkRequest, np.ndarray]]) -> None:
        """Route arriving 3D bricks to the correct channel slot."""
        by_channel: dict[int, list[tuple[ChunkRequest, np.ndarray]]] = {}
        for request, data in batch:
            if request.slice_request_id != self._current_slice_request_id_3d:
                continue
            ch_idx = int(request.axis_selections[self._channel_axis])
            by_channel.setdefault(ch_idx, []).append((request, data))

        for ch_idx, sub_batch in by_channel.items():
            slot_idx = self._channel_to_slot.get(ch_idx)
            if slot_idx is None:
                continue
            self._slots[slot_idx].on_data_ready(sub_batch)

    def on_data_ready_2d(self, batch: list[tuple[ChunkRequest, np.ndarray]]) -> None:
        """Route arriving 2D tiles to the correct channel slot."""
        # DEBUG print #3
        accepted: list[tuple[ChunkRequest, np.ndarray]] = []
        rejected = 0
        for request, data in batch:
            if request.slice_request_id != self._current_slice_request_id_2d:
                rejected += 1
            else:
                accepted.append((request, data))
        ch_counts: dict[int, int] = {}
        for request, _ in accepted:
            ch = int(request.axis_selections[self._channel_axis])
            ch_counts[ch] = ch_counts.get(ch, 0) + 1
        print(
            f"[MULTI-READY-2D] batch={len(batch)} accepted={len(accepted)} "
            f"guard_rejected={rejected} by_channel={ch_counts}"
        )

        by_channel: dict[int, list[tuple[ChunkRequest, np.ndarray]]] = {}
        for request, data in accepted:
            ch_idx = int(request.axis_selections[self._channel_axis])
            by_channel.setdefault(ch_idx, []).append((request, data))

        for ch_idx, sub_batch in by_channel.items():
            slot_idx = self._channel_to_slot.get(ch_idx)
            if slot_idx is None:
                continue
            self._slots[slot_idx].on_data_ready_2d(sub_batch)

    # ------------------------------------------------------------------
    # Standard event handlers
    # ------------------------------------------------------------------

    def on_transform_changed(self, event: TransformChangedEvent) -> None:
        """Update stored transform on all slots."""
        self._transform = event.transform
        for slot in self._slots:
            slot.on_transform_changed(event)

    def on_visibility_changed(self, event: VisualVisibilityChangedEvent) -> None:
        """Toggle wrapper Group visibility."""
        for g in (self._group_2d, self._group_3d):
            if g is not None:
                g.visible = event.visible

    def on_aabb_changed(self, event: AABBChangedEvent) -> None:
        """Propagate AABB toggle to all slots."""
        for slot in self._slots:
            slot.on_aabb_changed(event)

    def tick(self) -> None:
        """No per-frame state to advance."""
