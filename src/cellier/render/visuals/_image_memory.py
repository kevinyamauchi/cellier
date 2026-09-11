# src/cellier/v2/render/visuals/_image_memory.py
from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx

from cellier.data.image._image_requests import ChunkRequest
from cellier.render._spaces import RenderSpaces, node_matrix
from cellier.render.shaders._image_volume import IMAGE_VOLUME_MATERIALS
from cellier.render.visuals._pick import memory_image_data_coordinate
from cellier.render.visuals._slicing import (
    axis_selections_from_box,
)

if TYPE_CHECKING:
    from cellier._state import DimsState
    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.events._events import (
        AABBChangedEvent,
        AppearanceChangedEvent,
        PickWriteChangedEvent,
        TransformChangedEvent,
        VisualVisibilityChangedEvent,
    )
    from cellier.transform import (
        AffineTransform,
        RegionSelection,
        WorldCoordinateSystem,
    )
    from cellier.visuals._image_memory import ImageVisual


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_colormap(color_map) -> gfx.TextureMap:
    """Convert a cmap Colormap (or name string) to a pygfx TextureMap.

    Appearance mutations driven from the GUI can arrive as a colormap *name*
    string (e.g. the anywidget panel's Unicode trait), and psygnal emits the
    raw assigned value to bus subscribers, so the value reaching this consumer
    may be a ``str`` rather than a ``cmap.Colormap``.  Coerce it here.
    """
    from cmap import Colormap

    if not isinstance(color_map, Colormap):
        color_map = Colormap(color_map)
    return color_map.to_pygfx(N=256)


# Map InMemoryImageAppearance.render_mode -> volume material class.
#
# These are cellier's own subclasses, not pygfx's: the shader behind them
# writes the ``normal`` render target the ambient occlusion pass prefers,
# and fixes the iso branch's depth, which upstream computes as if the
# volume sat at the origin.  See
# ``cellier.render.shaders._image_volume``.
_VOLUME_MATERIALS: dict[str, type] = IMAGE_VOLUME_MATERIALS


def _make_volume_material(appearance, colormap, pick_write: bool):
    """Build the pygfx volume material for an ``InMemoryImageAppearance``.

    Selects the material class from ``appearance.render_mode`` and applies the
    shared appearance settings.  The ISO material additionally receives
    ``iso_threshold``.

    Parameters
    ----------
    appearance : InMemoryImageAppearance
        The appearance providing ``render_mode``, ``clim``, ``interpolation``,
        and ``iso_threshold``.
    colormap : gfx.TextureMap
        Pre-built colormap texture.
    pick_write : bool
        Whether the material writes to the pick buffer.

    Returns
    -------
    gfx.VolumeRayMaterial
        A configured volume material.
    """
    material_cls = _VOLUME_MATERIALS[appearance.render_mode]
    material = material_cls(
        clim=appearance.clim,
        map=colormap,
        interpolation=appearance.interpolation,
        pick_write=pick_write,
    )
    if appearance.render_mode == "iso":
        material.threshold = appearance.iso_threshold
    return material


def _pin_bounding_box(node, box_min: list[float], box_max: list[float]) -> None:
    """Override *node*'s ``get_bounding_box`` to report a fixed local box.

    The in-memory image/volume nodes start with a tiny placeholder texture
    (replaced on first ``on_data_ready[_2d]``), so the standard pygfx bounding
    box -- derived from the texture size -- would be 1- or 2-voxel and break
    camera fitting before any data is loaded.  Pinning the box to the full data
    footprint in local voxel space (``[-0.5, N-0.5]`` per axis) lets
    ``fit_camera`` frame the data correctly from the start, with no large
    upfront GPU allocation.

    The override is applied to the instance (rather than via a subclass) so the
    node is still built from the current ``gfx.Image`` / ``gfx.Volume`` symbol,
    which tests replace with a mock; a subclass would bind to the real pygfx
    class at import time and bypass that mock.
    """
    box = np.array([box_min, box_max], dtype=np.float32)
    node.get_bounding_box = lambda: box


def _box_wireframe_positions(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """Return (24, 3) float32 positions for a 3D box wireframe (12 edges x 2 pts)."""
    x0, y0, z0 = float(box_min[0]), float(box_min[1]), float(box_min[2])
    x1, y1, z1 = float(box_max[0]), float(box_max[1]), float(box_max[2])
    return np.array(
        [
            # bottom face
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y1, z0],
            [x0, y0, z0],
            # top face
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x1, y1, z1],
            [x0, y1, z1],
            [x0, y1, z1],
            [x0, y0, z1],
            # verticals
            [x0, y0, z0],
            [x0, y0, z1],
            [x1, y0, z0],
            [x1, y0, z1],
            [x1, y1, z0],
            [x1, y1, z1],
            [x0, y1, z0],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )


def _rect_wireframe_positions(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """Return (8, 3) float32 positions for a 2D rect wireframe (4 edges x 2 pts)."""
    x0, y0 = float(box_min[0]), float(box_min[1])
    x1, y1 = float(box_max[0]), float(box_max[1])
    return np.array(
        [
            [x0, y0, 0.0],
            [x1, y0, 0.0],
            [x1, y0, 0.0],
            [x1, y1, 0.0],
            [x1, y1, 0.0],
            [x0, y1, 0.0],
            [x0, y1, 0.0],
            [x0, y0, 0.0],
        ],
        dtype=np.float32,
    )


def _make_aabb_line(
    positions: np.ndarray, color: str, line_width: float = 2.0
) -> gfx.Line:
    """Create a gfx.Line for an AABB wireframe (initially invisible)."""
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineSegmentMaterial(color=color, thickness=line_width),
    )
    line.visible = False
    return line


def _plan_from_region(
    selection: RegionSelection,
    transform: AffineTransform,
    world: WorldCoordinateSystem,
    store_shape: tuple[int, ...],
) -> tuple[tuple[int | tuple[int, int], ...], dict[int, float]]:
    """Pull a world-space selection into voxel space and assemble the request.

    Design 3.7 steps 3 and 4.  ``imap_region`` is ``A^T`` on the normals and
    ``d - n . t`` on the offsets: no matrix inverse, no ``select_axes``, and no
    zero-filled probe point.  That last one is the reason to prefer it -- the
    old path invented coordinates for the displayed axes, wrote the slice
    positions into a zero vector and inverted, which is right for a diagonal
    transform and an arbitrary unstated choice for anything else.

    Parameters
    ----------
    selection : RegionSelection
        The region this canvas is showing, in world coordinates.
    transform : AffineTransform
        The visual's ``data -> world`` transform.
    world : WorldCoordinateSystem
        The transform's output system.  Needed to resolve its
        ``broadcast_axes``, which are stored as ids while the arithmetic
        wants indices.
    store_shape : tuple[int, ...]
        The store's shape, one entry per data axis.

    Returns
    -------
    tuple
        ``(axis_selections, collapsed_indices)`` -- the request's per-axis
        selection, and the voxel index of each axis that collapsed, which the
        node matrix needs (design 3.9).
    """
    data_region = transform.imap_region(selection.region, world).simplify()
    box = data_region.bounding_box()
    axis_selections = axis_selections_from_box(box, store_shape)
    collapsed = {
        axis: float(value)
        for axis, value in enumerate(axis_selections)
        if not isinstance(value, tuple)
    }
    return axis_selections, collapsed


# ---------------------------------------------------------------------------
# GFXImageMemoryVisual
# ---------------------------------------------------------------------------


class GFXImageMemoryVisual:
    """Render-layer visual for one ``ImageVisual`` backed by ``ImageMemoryStore``.

    Owns a single pygfx node -- either ``gfx.Image`` (render_mode=="2d") or
    ``gfx.Volume`` (render_mode=="3d") -- with a small placeholder texture that
    is replaced on the first ``on_data_ready[_2d]`` call.  The node's bounding
    box is pinned to the full data extent (see :func:`_pin_bounding_box`) so
    camera fitting works before any data is loaded.

    There is no brick cache, no LUT indirection, and no LOD selection. Every
    reslice produces exactly one ``ChunkRequest`` for the full slice / volume.
    The node geometry is replaced on every commit.

    Parameters
    ----------
    visual_model : ImageVisual
        Associated model-layer visual. Provides the initial appearance.
    data_store : ImageMemoryStore
        The backing data store. Used to query shape in planning methods.
    render_modes : set[str]
        Which nodes to build: ``{"2d"}``, ``{"3d"}``, or ``{"2d", "3d"}``.
    """

    cancellable: bool = True

    def __init__(
        self,
        visual_model: ImageVisual,
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
        self.render_modes: set[str] = render_modes
        self._data_store = data_store

        # The data -> world transform.  There is no coordinate-system-less
        # identity to fall back on (D18), so a visual reaching the render layer
        # without one cannot be placed until the controller supplies it.
        self._transform: AffineTransform | None = transform

        # The systems this visual's geometry is placed with, pushed by the
        # controller when displayed_axes changes.  ``None`` until the scene has
        # a canvas.
        self._spaces: RenderSpaces | None = None

        # Track displayed axes for lazy node matrix updates (Option C).
        self._last_displayed_axes: tuple[int, ...] | None = None
        # The collapsed voxel index per dropped data axis, refreshed on every
        # planned request.  The node matrix needs them because a collapsed
        # axis's index enters the translation (design 3.9); it contributes
        # nothing to the displayed rows of a block-diagonal transform, which is
        # why ``select_axes`` got away with dropping it.
        self._collapsed_indices: dict[int, float] = {}

        # Data-ready flags: AABB visibility is suppressed until real data
        # has been committed (prevents showing wrong placeholder geometry).
        self._data_ready_2d: bool = False
        self._data_ready_3d: bool = False

        # Cache initial AABB params for use when data first arrives.
        self._aabb_enabled: bool = visual_model.aabb.enabled
        self._aabb_color: str = visual_model.aabb.color
        self._aabb_line_width: float = visual_model.aabb.line_width

        appearance = visual_model.appearance
        colormap = _make_colormap(appearance.color_map)

        # Cached state needed to rebuild the 3D volume material when the
        # render_mode changes (pygfx materials are not switchable in place).
        self._colormap = colormap
        self._pick_write: bool = visual_model.pick_write
        self._render_mode: str = appearance.render_mode
        self._iso_threshold: float = appearance.iso_threshold

        self.node_2d: gfx.Group | None = None
        self._inner_node_2d: gfx.Image | None = None
        self._aabb_line_2d: gfx.Line | None = None

        self.node_3d: gfx.Group | None = None
        self._inner_node_3d: gfx.Volume | None = None
        self._aabb_line_3d: gfx.Line | None = None

        if "2d" in render_modes:
            # Placeholder 1x1 texture -- replaced on first on_data_ready_2d.
            # The node's bounding box is pinned to the full data extent (see
            # _pin_bounding_box) so fit_camera works before any data is loaded.
            h, w = data_store.shape[-2], data_store.shape[-1]
            placeholder = np.zeros((1, 1, 1), dtype=np.float32)
            tex = gfx.Texture(placeholder, dim=2, format="1xf4")
            self._inner_node_2d = gfx.Image(
                gfx.Geometry(grid=tex),
                gfx.ImageBasicMaterial(
                    clim=appearance.clim,
                    map=colormap,
                    interpolation=appearance.interpolation,
                    pick_write=visual_model.pick_write,
                ),
            )
            _pin_bounding_box(
                self._inner_node_2d, [-0.5, -0.5, 0.0], [w - 0.5, h - 0.5, 0.0]
            )
            # AABB placeholder; geometry replaced on first on_data_ready_2d.
            placeholder_positions = _rect_wireframe_positions(np.zeros(2), np.ones(2))
            self._aabb_line_2d = _make_aabb_line(
                placeholder_positions, self._aabb_color, self._aabb_line_width
            )
            self.node_2d = gfx.Group()
            self.node_2d.add(self._inner_node_2d)
            self.node_2d.add(self._aabb_line_2d)

        if "3d" in render_modes:
            # Placeholder 2x2x2 texture -- replaced on first on_data_ready.
            # The node's bounding box is pinned to the full data extent (see
            # _pin_bounding_box) so fit_camera works before any data is loaded.
            d, h, w = data_store.shape[-3], data_store.shape[-2], data_store.shape[-1]
            placeholder = np.zeros((2, 2, 2), dtype=np.float32)
            tex = gfx.Texture(placeholder, dim=3, format="1xf4")
            self._inner_node_3d = gfx.Volume(
                gfx.Geometry(grid=tex),
                _make_volume_material(appearance, colormap, visual_model.pick_write),
            )
            _pin_bounding_box(
                self._inner_node_3d,
                [-0.5, -0.5, -0.5],
                [w - 0.5, h - 0.5, d - 0.5],
            )
            # AABB placeholder; geometry replaced on first on_data_ready.
            placeholder_positions = _box_wireframe_positions(np.zeros(3), np.ones(3))
            self._aabb_line_3d = _make_aabb_line(
                placeholder_positions, self._aabb_color, self._aabb_line_width
            )
            self.node_3d = gfx.Group()
            self.node_3d.add(self._inner_node_3d)
            self.node_3d.add(self._aabb_line_3d)

        if self.node_2d is not None:
            self.node_2d.render_order = visual_model.appearance.render_order
        if self.node_3d is not None:
            self.node_3d.render_order = visual_model.appearance.render_order

        for inner in (self._inner_node_2d, self._inner_node_3d):
            if inner is not None:
                inner.material.opacity = appearance.opacity
                inner.material.depth_test = appearance.depth_test
                inner.material.depth_write = appearance.depth_write
                inner.material.depth_compare = appearance.depth_compare
                inner.material.alpha_mode = appearance.transparency_mode

        for node in (self.node_2d, self.node_3d):
            if node is not None:
                node.visible = appearance.visible

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

    def set_render_spaces(self, spaces: RenderSpaces | None) -> None:
        """Receive the coordinate systems this visual is placed with.

        Called by the controller when ``displayed_axes`` changes -- a pure
        reorder included -- and when the first canvas gives the scene a
        rendered system.
        """
        self._spaces = spaces
        if spaces is not None and self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)

    def _update_node_matrix(self, displayed_axes: tuple[int, ...]) -> None:
        """Recompute and apply the pygfx node matrix for *displayed_axes*.

        Called from ``build_slice_request*`` when displayed axes change, and
        from ``on_transform_changed`` when the transform is updated.  The
        matrix is set on the Group nodes so all children (inner image/volume
        and AABB line) inherit the same transform.

        The matrix is the composition of design 3.9,
        ``visual -> data -> world -> rendered``, rather than the
        ``select_axes`` sub-block it used to be.  The two agree exactly
        whenever the linear block is block-diagonal with respect to the
        displayed set, which is every transform shipping today; they diverge
        on one with a cross-term, where ``select_axes`` was silently wrong.

        A no-op while the visual has no systems yet: nothing is drawn before
        the scene has a canvas, and ``set_render_spaces`` recomputes.
        """
        self._last_displayed_axes = displayed_axes
        if self._spaces is None or self._transform is None:
            return
        m = node_matrix(
            self._spaces, self._transform, self._collapsed_indices_or_origin()
        )
        if self.node_3d is not None:
            self.node_3d.local.matrix = m
        if self.node_2d is not None:
            self.node_2d.local.matrix = m

    def _collapsed_indices_or_origin(self) -> dict[int, float]:
        """The last planned collapsed indices, defaulting to the origin plane.

        Before the first slice request nothing has said where the collapsed
        axes sit.  Zero is the origin plane rather than a guess at the user's
        slice, and it is corrected the moment a request is planned -- which
        happens before anything is drawn.
        """
        return {
            axis: float(self._collapsed_indices.get(axis, 0.0))
            for axis in self._spaces.collapsed_axes
        }

    # ------------------------------------------------------------------
    # Node selection
    # ------------------------------------------------------------------

    def get_node_for_dims(self, displayed_axes: tuple[int, ...]) -> gfx.Group | None:
        """Return the pre-built node appropriate for *displayed_axes*.

        Also eagerly updates the node matrix so the transform is correct
        before the node is placed in the scene by ``SceneManager.swap_node``.

        Parameters
        ----------
        displayed_axes : tuple[int, ...]
            The new set of displayed axes.

        Returns
        -------
        gfx.Group or None
            ``node_3d`` when ``len(displayed_axes) == 3``,
            ``node_2d`` otherwise.  Returns ``None`` if the required node
            was not built (e.g. 3D axes requested but only ``"2d"`` mode
            was specified).
        """
        if len(displayed_axes) == 3:
            node = self.node_3d
        else:
            node = self.node_2d

        # Eagerly update the node matrix so the transform is already applied
        # when the node enters the scene, rather than waiting for the next
        # build_slice_request call.
        if displayed_axes != self._last_displayed_axes:
            self._update_node_matrix(displayed_axes)

        return node

    # ── GFXVisual protocol ──────────────────────────────────────────────

    def has_node(self, mode: str) -> bool:
        return self.node_3d is not None if mode == "3d" else self.node_2d is not None

    def get_node(self, mode: str) -> gfx.Group | None:
        return self.get_node_for_dims(
            tuple(range(3)) if mode == "3d" else tuple(range(2))
        )

    def build_node(
        self, mode, visual_model, displayed_axes, level_shapes, level_transforms
    ):
        return self.get_node_for_dims(displayed_axes)

    def rebuild_node_geometry(
        self, mode, displayed_axes, level_shapes, level_transforms
    ):
        return self.get_node_for_dims(displayed_axes)

    def on_stacked_axes_changed(self, stacked_axes: tuple[int, ...]) -> None:
        pass

    # ------------------------------------------------------------------
    # Planning -- build ChunkRequests (synchronous, < 1 ms)
    # ------------------------------------------------------------------

    def _axis_selections(
        self, dims_state: DimsState, selection: RegionSelection | None
    ) -> tuple[int | tuple[int, int], ...]:
        """Plan one request's per-axis selection, and record where it collapsed.

        The ``RegionSelection`` the controller built is the only path: it
        pulls the whole selected region back through the transform in one
        operation that needs no inverse, and it extends to a slab, a viewport
        crop or an oblique plane by changing only the region.
        """
        shape = self._data_store.shape
        if selection is None or self._spaces is None:
            raise RuntimeError(
                "This visual has no region to plan from: either it has not "
                "been placed in a world or the reslicing request carried no "
                "selection.  Until v1 was retired this fell back to reading "
                "``dims_state.slice_indices`` as world positions."
            )
        axis_selections, collapsed = _plan_from_region(
            selection, self._transform, self._spaces.world, shape
        )
        self._collapsed_indices = collapsed
        return axis_selections

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
        selection: RegionSelection | None = None,
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
        selection : RegionSelection or None
            The region this canvas is showing, in world coordinates.  When
            given it decides what is fetched; ``dims_state`` is the fallback
            for a headlessly driven visual.

        Returns
        -------
        list[ChunkRequest]
            Always contains exactly one element.
        """
        displayed = dims_state.selection.displayed_axes
        axis_selections = self._axis_selections(dims_state, selection)
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)
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
        fov_y_rad: float,
        screen_height_px: float,
        lod_bias: float = 1.0,
        dims_state: DimsState | None = None,
        force_level: int | None = None,
        selection: RegionSelection | None = None,
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
        fov_y_rad : float
            Unused. Accepted for interface compatibility.
        screen_height_px : float
            Unused. Accepted for interface compatibility.
        lod_bias : float
            Unused. Accepted for interface compatibility.
        dims_state : DimsState or None
            Current dimension state. If ``None`` (e.g. during headless
            tests), all axes are treated as displayed.
        force_level : int or None
            Unused. Accepted for interface compatibility.
        selection : RegionSelection or None
            The region this canvas is showing, in world coordinates.  When
            given it decides what is fetched; ``dims_state`` is the fallback
            for a headlessly driven visual.

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
            axis_selections = self._axis_selections(dims_state, selection)
            if displayed != self._last_displayed_axes:
                self._update_node_matrix(displayed)

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
        if not batch or self._inner_node_3d is None:
            return

        _request, data = batch[0]

        # No transpose: pygfx maps a numpy texture's *last* axis to texture-x,
        # so (D, H, W) = (z, y, x) already lands as local (x=W, y=H, z=D) -- the
        # desired data(z, y, x) -> world(x, y, z) mapping. This mirrors the 2-D
        # path, which also uploads without transposing. (A previous ``data.T``
        # double-reversed the axes -- pygfx already reverses once -- which
        # transposed data-x and data-z in the rendered volume.)
        data_wgpu = np.ascontiguousarray(data)
        tex = gfx.Texture(data_wgpu, dim=3, format="1xf4")
        self._inner_node_3d.geometry = gfx.Geometry(grid=tex)

        # On first real data: rebuild AABB geometry from true shape and
        # apply the pending aabb.enabled state.
        if not self._data_ready_3d and self._aabb_line_3d is not None:
            d, h, w = data.shape  # data is (D, H, W)
            # pygfx voxel convention: voxel i center at i, edges at i±0.5.
            # Volume of shape N spans [-0.5, N-0.5] in local space.
            positions = _box_wireframe_positions(
                np.array([-0.5, -0.5, -0.5]),
                np.array([w - 0.5, h - 0.5, d - 0.5]),
            )
            self._aabb_line_3d.geometry = gfx.Geometry(positions=positions)
            self._data_ready_3d = True
            self._aabb_line_3d.visible = self._aabb_enabled

    def on_data_ready_2d(self, batch: list[tuple[ChunkRequest, np.ndarray]]) -> None:
        """Upload a 2-D slice to the pygfx Image node.

        Called on the main thread by ``SliceCoordinator`` after the
        ``AsyncSlicer`` completes the read.

        Parameters
        ----------
        batch : list of (ChunkRequest, np.ndarray)
            Always contains exactly one element for this visual type.
        """
        if not batch or self._inner_node_2d is None:
            return

        _request, data = batch[0]

        # pygfx Image expects (H, W, 1) -- add channel dim, no transpose.
        data_wgpu = np.ascontiguousarray(data[:, :, np.newaxis])
        tex = gfx.Texture(data_wgpu, dim=2, format="1xf4")
        self._inner_node_2d.geometry = gfx.Geometry(grid=tex)

        # On first real data: rebuild AABB rect geometry from true shape and
        # apply the pending aabb.enabled state.
        if not self._data_ready_2d and self._aabb_line_2d is not None:
            h, w = data.shape  # data is (H, W)
            # pygfx voxel convention: pixel i center at i, edges at i±0.5.
            # Image of shape N spans [-0.5, N-0.5] in local space.
            positions = _rect_wireframe_positions(
                np.array([-0.5, -0.5]),
                np.array([w - 0.5, h - 0.5]),
            )
            self._aabb_line_2d.geometry = gfx.Geometry(positions=positions)
            self._data_ready_2d = True
            self._aabb_line_2d.visible = self._aabb_enabled

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

    def _rebuild_volume_material(self) -> None:
        """Swap the 3D volume material to match the current ``render_mode``.

        pygfx volume materials cannot change their ray-cast mode in place, so a
        ``render_mode`` change requires constructing a fresh material. The
        current material's dynamic settings (clim, interpolation, opacity,
        depth, alpha) are carried over so only the rendering mode changes.
        """
        if self._inner_node_3d is None:
            return
        old = self._inner_node_3d.material
        material_cls = _VOLUME_MATERIALS[self._render_mode]
        material = material_cls(
            clim=old.clim,
            map=self._colormap,
            interpolation=old.interpolation,
            pick_write=self._pick_write,
        )
        material.opacity = old.opacity
        material.depth_test = old.depth_test
        material.depth_write = old.depth_write
        material.depth_compare = old.depth_compare
        material.alpha_mode = old.alpha_mode
        if self._render_mode == "iso":
            material.threshold = self._iso_threshold
        self._inner_node_3d.material = material

    def on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        """Apply a pure GPU-side appearance update (no reslice needed).

        Handles ``clim``, ``color_map``, ``interpolation``, and -- for the 3D
        view -- ``render_mode`` and ``iso_threshold``. Unrecognised field
        names are silently ignored.

        Parameters
        ----------
        event : AppearanceChangedEvent
            Carries ``field_name`` and ``new_value``.
        """
        # render_mode swaps the whole 3D material (pygfx materials cannot be
        # switched in place); handle it separately, before the per-node loop.
        if event.field_name == "render_mode":
            self._render_mode = event.new_value
            self._rebuild_volume_material()
            return
        if event.field_name == "iso_threshold":
            self._iso_threshold = event.new_value
            if self._inner_node_3d is not None and self._render_mode == "iso":
                self._inner_node_3d.material.threshold = event.new_value
            return

        for inner in (self._inner_node_2d, self._inner_node_3d):
            if inner is None:
                continue
            material = inner.material
            if event.field_name == "clim":
                material.clim = event.new_value
            elif event.field_name == "color_map":
                self._colormap = _make_colormap(event.new_value)
                material.map = self._colormap
            elif event.field_name == "interpolation":
                material.interpolation = event.new_value
            elif event.field_name == "opacity":
                material.opacity = event.new_value
            elif event.field_name in ("depth_test", "depth_write", "depth_compare"):
                setattr(material, event.field_name, event.new_value)
            elif event.field_name == "transparency_mode":
                material.alpha_mode = event.new_value
        if event.field_name == "render_order":
            if self.node_2d is not None:
                self.node_2d.render_order = event.new_value
            if self.node_3d is not None:
                self.node_3d.render_order = event.new_value
        # "visible" is handled by on_visibility_changed; ignore here.

    def on_visibility_changed(self, event: VisualVisibilityChangedEvent) -> None:
        """Toggle Group node visibility.

        Parameters
        ----------
        event : VisualVisibilityChangedEvent
            Carries ``visible`` bool.
        """
        for node in (self.node_2d, self.node_3d):
            if node is not None:
                node.visible = event.visible

    def pick_collapsed_indices(self) -> dict[int, int] | None:
        """The level-0 planes this visual last drew, per collapsed data axis.

        Read off the plan rather than recomputed from the dims state, so a
        pick reports the slice that is actually on screen even while a
        reslice is in flight.  ``None`` before the first one.

        Returns
        -------
        dict[int, int] or None
            Data axis to voxel index, for collapsed axes only.
        """
        if not self._collapsed_indices:
            return None
        return {axis: int(value) for axis, value in self._collapsed_indices.items()}

    def pick_data_coordinate(
        self, hit_object, pick_info: dict
    ) -> tuple[float, ...] | None:
        """Level-0 data coordinate of a pick on this visual (displayed axes).

        The texture holds the displayed data slice/volume directly, so pygfx's
        ``index`` is already the data index; see
        :func:`cellier.render.visuals._pick.memory_image_data_coordinate`.
        """
        ndim = 3 if isinstance(hit_object, gfx.Volume) else 2
        return memory_image_data_coordinate(pick_info, ndim)

    def on_pick_write_changed(self, event: PickWriteChangedEvent) -> None:
        """Update pick_write on all inner node materials."""
        self._pick_write = event.pick_write
        for inner in (self._inner_node_2d, self._inner_node_3d):
            if inner is not None:
                inner.material.pick_write = event.pick_write

    def on_aabb_changed(self, event: AABBChangedEvent) -> None:
        """Apply an AABB parameter change.

        ``enabled`` toggles AABB line visibility (guarded by data-ready
        flags so the line cannot appear before real geometry is in place).
        ``color`` updates the line material.

        Parameters
        ----------
        event : AABBChangedEvent
            Carries ``field_name`` and ``new_value``.
        """
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
        """Called once per rendered frame. No per-frame state to advance.

        If any per-frame state, implement it here (e.g., temporal jitter seed).
        """
        pass
