# src/cellier/v2/render/visuals/_label_memory.py
from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx

# Import the shader modules to trigger @register_wgpu_render_function.
import cellier.v2.render.shaders._label_image
import cellier.v2.render.shaders._label_volume  # noqa: F401
from cellier.v2._state import AxisAlignedSelectionState, DimsState
from cellier.v2.data.image._image_requests import ChunkRequest
from cellier.v2.render.shaders._label_colormap import (
    build_direct_lut_textures,
    build_label_params_buffer,
)
from cellier.v2.render.shaders._label_image import LabelImageMaterial
from cellier.v2.render.shaders._label_volume import LabelVolumeMaterial
from cellier.v2.render.visuals._image_memory import (
    _box_wireframe_positions,
    _build_axis_selections,
    _make_aabb_line,
    _pygfx_matrix,
    _rect_wireframe_positions,
    _transform_slice_indices,
)

if TYPE_CHECKING:
    from cellier.v2.data.label._label_memory_store import LabelMemoryStore
    from cellier.v2.events._events import (
        AABBChangedEvent,
        AppearanceChangedEvent,
        TransformChangedEvent,
        VisualVisibilityChangedEvent,
    )
    from cellier.v2.transform import AffineTransform
    from cellier.v2.visuals._label_memory import LabelMemoryVisual


def _make_placeholder_label_params() -> gfx.Buffer:
    return build_label_params_buffer(background_label=0, salt=0, n_entries=0)


class GFXLabelMemoryVisual:
    """Render-layer visual for a LabelMemoryVisual backed by LabelMemoryStore.

    Owns gfx.Image (2D) and/or gfx.Volume (3D) nodes with custom label shaders.
    No brick cache; every reslice uploads the full slice or volume.

    Parameters
    ----------
    visual_model : LabelMemoryVisual
        Associated model-layer visual.
    data_store : LabelMemoryStore
        Backing data store for shape queries.
    render_modes : set[str]
        Which nodes to build: ``{"2d"}``, ``{"3d"}``, or ``{"2d", "3d"}``.
    """

    cancellable: bool = True

    def __init__(
        self,
        visual_model: LabelMemoryVisual,
        data_store: LabelMemoryStore,
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
        self.render_modes: set[str] = render_modes
        self._data_store = data_store

        if transform is None:
            from cellier.v2.transform import AffineTransform as _AT

            transform = _AT.identity(ndim=data_store.ndim)
        elif transform.ndim < data_store.ndim:
            transform = transform.expand_dims(data_store.ndim)
        self._transform: AffineTransform = transform

        self._last_displayed_axes: tuple[int, ...] | None = None
        self._data_ready_2d: bool = False
        self._data_ready_3d: bool = False

        self._aabb_enabled: bool = visual_model.aabb.enabled
        self._aabb_color: str = visual_model.aabb.color
        self._aabb_line_width: float = visual_model.aabb.line_width

        appearance = visual_model.appearance
        self._background_label: int = appearance.background_label

        # Build initial GPU resources from appearance.
        label_params_buf = build_label_params_buffer(
            background_label=appearance.background_label,
            salt=appearance.salt,
            n_entries=0,
        )
        keys_tex = colors_tex = None
        n_entries = 0
        if appearance.colormap_mode == "direct" and appearance.color_dict:
            keys_tex, colors_tex, n_entries = build_direct_lut_textures(
                appearance.color_dict
            )
            label_params_buf = build_label_params_buffer(
                background_label=appearance.background_label,
                salt=appearance.salt,
                n_entries=n_entries,
            )

        self._label_params_buf: gfx.Buffer = label_params_buf
        self._keys_tex = keys_tex
        self._colors_tex = colors_tex
        self._n_entries: int = n_entries

        self.node_2d: gfx.Group | None = None
        self._inner_node_2d: gfx.Image | None = None
        self._aabb_line_2d: gfx.Line | None = None

        self.node_3d: gfx.Group | None = None
        self._inner_node_3d: gfx.Volume | None = None
        self._aabb_line_3d: gfx.Line | None = None

        if "2d" in render_modes:
            placeholder = np.zeros((1, 1, 1), dtype=np.int32)
            tex = gfx.Texture(placeholder, dim=2, format="1xi4")
            mat2d = LabelImageMaterial(
                background_label=appearance.background_label,
                colormap_mode=appearance.colormap_mode,
                salt=appearance.salt,
                label_keys_texture=keys_tex,
                label_colors_texture=colors_tex,
                n_entries=n_entries,
                label_params_buffer=label_params_buf,
                opacity=appearance.opacity,
                pick_write=visual_model.pick_write,
            )
            self._inner_node_2d = gfx.Image(gfx.Geometry(grid=tex), mat2d)
            placeholder_pos = _rect_wireframe_positions(np.zeros(2), np.ones(2))
            self._aabb_line_2d = _make_aabb_line(
                placeholder_pos, self._aabb_color, self._aabb_line_width
            )
            self.node_2d = gfx.Group()
            self.node_2d.add(self._inner_node_2d)
            self.node_2d.add(self._aabb_line_2d)

        if "3d" in render_modes:
            placeholder = np.zeros((2, 2, 2), dtype=np.int32)
            tex = gfx.Texture(placeholder, dim=3, format="1xi4")
            mat3d = LabelVolumeMaterial(
                background_label=appearance.background_label,
                colormap_mode=appearance.colormap_mode,
                salt=appearance.salt,
                render_mode=appearance.render_mode,
                label_keys_texture=keys_tex,
                label_colors_texture=colors_tex,
                n_entries=n_entries,
                label_params_buffer=label_params_buf,
            )
            self._inner_node_3d = gfx.Volume(gfx.Geometry(grid=tex), mat3d)
            placeholder_pos = _box_wireframe_positions(np.zeros(3), np.ones(3))
            self._aabb_line_3d = _make_aabb_line(
                placeholder_pos, self._aabb_color, self._aabb_line_width
            )
            self.node_3d = gfx.Group()
            self.node_3d.add(self._inner_node_3d)
            self.node_3d.add(self._aabb_line_3d)

        if self.node_2d is not None:
            self.node_2d.render_order = appearance.render_order
        if self.node_3d is not None:
            self.node_3d.render_order = appearance.render_order

        for inner in (self._inner_node_2d, self._inner_node_3d):
            if inner is not None:
                inner.material.depth_test = appearance.depth_test
                inner.material.depth_write = appearance.depth_write
                inner.material.depth_compare = appearance.depth_compare
                inner.material.alpha_mode = appearance.transparency_mode

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_levels(self) -> int:
        return 1

    # ------------------------------------------------------------------
    # Cancellation stubs
    # ------------------------------------------------------------------

    def cancel_pending(self) -> None:
        pass

    def cancel_pending_2d(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Node matrix
    # ------------------------------------------------------------------

    def _update_node_matrix(self, displayed_axes: tuple[int, ...]) -> None:
        self._last_displayed_axes = displayed_axes
        sub = self._transform.select_axes(displayed_axes)
        m = _pygfx_matrix(sub)
        if self.node_3d is not None:
            self.node_3d.local.matrix = m
        if self.node_2d is not None:
            self.node_2d.local.matrix = m

    # ------------------------------------------------------------------
    # Node selection
    # ------------------------------------------------------------------

    def get_node_for_dims(self, displayed_axes: tuple[int, ...]) -> gfx.Group | None:
        if len(displayed_axes) == 3:
            node = self.node_3d
        else:
            node = self.node_2d
        if displayed_axes != self._last_displayed_axes:
            self._update_node_matrix(displayed_axes)
        return node

    # ------------------------------------------------------------------
    # Slice request planning
    # ------------------------------------------------------------------

    def build_slice_request_2d(
        self,
        camera_pos_world,
        viewport_width_px,
        world_width,
        view_min_world,
        view_max_world,
        dims_state: DimsState,
        lod_bias: float = 1.0,
        force_level: int | None = None,
        use_culling: bool = True,
    ) -> list[ChunkRequest]:
        displayed = dims_state.selection.displayed_axes
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)

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
        camera_pos_world,
        frustum_corners_world,
        fov_y_rad: float,
        screen_height_px: float,
        lod_bias: float = 1.0,
        dims_state: DimsState | None = None,
        force_level: int | None = None,
    ) -> list[ChunkRequest]:
        if dims_state is None:
            ndim = self._data_store.ndim
            axis_selections = tuple(
                (0, self._data_store.shape[ax]) for ax in range(ndim)
            )
        else:
            displayed = dims_state.selection.displayed_axes
            if displayed != self._last_displayed_axes:
                self._update_node_matrix(displayed)

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
    # Data commit callbacks
    # ------------------------------------------------------------------

    def on_data_ready(self, batch: list[tuple[ChunkRequest, np.ndarray]]) -> None:
        """Upload a 3-D int32 array to the pygfx Volume node."""
        if not batch or self._inner_node_3d is None:
            return
        _request, data = batch[0]

        # pygfx size_from_array for dim=3 maps numpy (D, H, W) → texture (W, H, D).
        # Pass data as-is so texture dims match _pygfx_matrix's (X, Y, Z) scale order.
        data_wgpu = np.ascontiguousarray(data.astype(np.int32))
        tex = gfx.Texture(data_wgpu, dim=3, format="1xi4")
        self._inner_node_3d.geometry = gfx.Geometry(grid=tex)

        if not self._data_ready_3d and self._aabb_line_3d is not None:
            d, h, w = data.shape
            positions = _box_wireframe_positions(
                np.array([-0.5, -0.5, -0.5]),
                np.array([w - 0.5, h - 0.5, d - 0.5]),
            )
            self._aabb_line_3d.geometry = gfx.Geometry(positions=positions)
            self._data_ready_3d = True
            self._aabb_line_3d.visible = self._aabb_enabled

    def on_data_ready_2d(self, batch: list[tuple[ChunkRequest, np.ndarray]]) -> None:
        """Upload a 2-D int32 slice to the pygfx Image node."""
        if not batch or self._inner_node_2d is None:
            return
        _request, data = batch[0]

        # pygfx Image expects (H, W, 1).
        data_wgpu = np.ascontiguousarray(data[:, :, np.newaxis].astype(np.int32))
        tex = gfx.Texture(data_wgpu, dim=2, format="1xi4")
        self._inner_node_2d.geometry = gfx.Geometry(grid=tex)

        if not self._data_ready_2d and self._aabb_line_2d is not None:
            h, w = data.shape
            positions = _rect_wireframe_positions(
                np.array([-0.5, -0.5]),
                np.array([w - 0.5, h - 0.5]),
            )
            self._aabb_line_2d.geometry = gfx.Geometry(positions=positions)
            self._data_ready_2d = True
            self._aabb_line_2d.visible = self._aabb_enabled

    # ------------------------------------------------------------------
    # Uniform buffer update helpers
    # ------------------------------------------------------------------

    def _update_label_params_uniform(
        self,
        background_label: int | None = None,
        salt: int | None = None,
        n_entries: int | None = None,
    ) -> None:
        """Write new values into the existing label_params buffer in place."""
        import numpy as np

        from cellier.v2.render.shaders._label_colormap import LABEL_PARAMS_DTYPE

        if background_label is not None:
            self._background_label = background_label
        buf = self._label_params_buf
        data = np.frombuffer(buf.data, dtype=LABEL_PARAMS_DTYPE).copy()
        if background_label is not None:
            data["background_label"] = np.int32(background_label)
        if salt is not None:
            data["salt"] = np.uint32(salt & 0xFFFFFFFF)
        if n_entries is not None:
            data["n_entries"] = np.uint32(n_entries)
            self._n_entries = n_entries
        buf.data[:] = data.tobytes()
        buf.update_range(0, 1)

    def _rebuild_lut(
        self, color_dict: dict[int, tuple[float, float, float, float]]
    ) -> None:
        """Rebuild LUT textures and update material texture references."""
        keys_tex, colors_tex, n_entries = build_direct_lut_textures(color_dict)
        self._keys_tex = keys_tex
        self._colors_tex = colors_tex
        self._n_entries = n_entries
        self._update_label_params_uniform(n_entries=n_entries)
        for inner in (self._inner_node_2d, self._inner_node_3d):
            if inner is not None:
                inner.material.label_keys_texture = keys_tex
                inner.material.label_colors_texture = colors_tex
                inner.material.n_entries = n_entries

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        fn = event.field_name
        val = event.new_value

        if fn == "opacity":
            for inner in (self._inner_node_2d, self._inner_node_3d):
                if inner is not None:
                    inner.material.opacity = val

        elif fn == "background_label":
            self._update_label_params_uniform(background_label=val)

        elif fn == "salt":
            self._update_label_params_uniform(salt=val)

        elif fn == "color_dict":
            self._rebuild_lut(val)

        elif fn in ("depth_test", "depth_write", "depth_compare"):
            for inner in (self._inner_node_2d, self._inner_node_3d):
                if inner is not None:
                    setattr(inner.material, fn, val)

        elif fn == "transparency_mode":
            for inner in (self._inner_node_2d, self._inner_node_3d):
                if inner is not None:
                    inner.material.alpha_mode = val

        elif fn == "render_order":
            if self.node_2d is not None:
                self.node_2d.render_order = val
            if self.node_3d is not None:
                self.node_3d.render_order = val

        elif fn in ("colormap_mode", "render_mode"):
            import warnings

            warnings.warn(
                f"LabelMemoryVisual.appearance.{fn} is frozen after construction. "
                "Create a new visual to change it.",
                stacklevel=2,
            )

    def on_transform_changed(self, event: TransformChangedEvent) -> None:
        self._transform = event.transform
        if self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)

    def on_visibility_changed(self, event: VisualVisibilityChangedEvent) -> None:
        for node in (self.node_2d, self.node_3d):
            if node is not None:
                node.visible = event.visible

    def on_aabb_changed(self, event: AABBChangedEvent) -> None:
        if event.field_name == "enabled":
            self._aabb_enabled = event.new_value
            if self._aabb_line_2d is not None:
                self._aabb_line_2d.visible = event.new_value and self._data_ready_2d
            if self._aabb_line_3d is not None:
                self._aabb_line_3d.visible = event.new_value and self._data_ready_3d
        elif event.field_name == "color":
            self._aabb_color = event.new_value
            for line in (self._aabb_line_2d, self._aabb_line_3d):
                if line is not None:
                    line.material.color = event.new_value
        elif event.field_name == "line_width":
            self._aabb_line_width = event.new_value
            for line in (self._aabb_line_2d, self._aabb_line_3d):
                if line is not None:
                    line.material.thickness = event.new_value

    def tick(self) -> None:
        pass
