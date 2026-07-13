"""Tests for the anywidget (notebook) front-end.

The whole module is skipped when ``rendercanvas.anywidget`` (and the
``anywidget`` package) is not importable, so CI without the optional extra
still passes.  These tests never construct a ``QApplication``.
"""

from __future__ import annotations

from uuid import uuid4

import pygfx as gfx
import pytest

rendercanvas_anywidget = pytest.importorskip("rendercanvas.anywidget")

from cellier.controller import CellierController  # noqa: E402
from cellier.render.canvas_view import CanvasView  # noqa: E402
from cellier.scene.dims import (  # noqa: E402
    AxisAlignedSelection,
    CoordinateSystem,
    DimsManager,
)
from cellier.scene.scene import Scene  # noqa: E402


def _make_scene(
    displayed_axes: tuple[int, ...],
    render_modes: set[str] | None = None,
) -> Scene:
    """Return a minimal Scene with the given displayed_axes and render_modes."""
    n_axes = max(displayed_axes) + 1
    axis_labels = tuple(f"axis_{i}" for i in range(n_axes))
    cs = CoordinateSystem(name="world", axis_labels=axis_labels)
    slice_indices = {i: 0 for i in range(n_axes) if i not in displayed_axes}
    dims = DimsManager(
        coordinate_system=cs,
        selection=AxisAlignedSelection(
            displayed_axes=displayed_axes,
            slice_indices=slice_indices,
        ),
    )
    resolved_modes = render_modes if render_modes is not None else {"2d", "3d"}
    return Scene(name="test_scene", dims=dims, render_modes=resolved_modes)


# ---------------------------------------------------------------------------
# Task 3 - canvas backend seam
# ---------------------------------------------------------------------------


def test_canvas_view_anywidget_creates_anywidget_canvas():
    """CanvasView(gui='anywidget') builds a rendercanvas anywidget canvas."""
    scene = gfx.Scene()
    canvas_view = CanvasView(
        uuid4(), uuid4(), get_scene_fn=lambda _sid: scene, dim="2d", gui="anywidget"
    )
    assert isinstance(canvas_view.widget, rendercanvas_anywidget.RenderCanvas)


def test_canvas_view_unknown_gui_raises():
    """An unknown gui value is rejected."""
    scene = gfx.Scene()
    with pytest.raises(ValueError, match="Unknown gui"):
        CanvasView(
            uuid4(), uuid4(), get_scene_fn=lambda _sid: scene, dim="2d", gui="tk"
        )


# ---------------------------------------------------------------------------
# Task 4 - gui threaded through the controller
# ---------------------------------------------------------------------------


def test_controller_gui_anywidget_creates_anywidget_canvas():
    """A gui='anywidget' controller builds anywidget canvases via add_canvas."""
    controller = CellierController(gui="anywidget")
    scene = controller.add_scene_model(_make_scene(displayed_axes=(1, 2)))

    controller.add_canvas(scene_id=scene.id)

    canvas_id = controller.get_canvas_ids(scene.id)[-1]
    canvas_view = controller.get_canvas_view(canvas_id)
    assert isinstance(canvas_view.widget, rendercanvas_anywidget.RenderCanvas)


def test_controller_default_gui_is_qt():
    """The controller defaults to the Qt gui."""
    controller = CellierController()
    assert controller._gui == "qt"


# ---------------------------------------------------------------------------
# Task 5 - Viewer / OrthoViewer accept gui
# ---------------------------------------------------------------------------


def test_viewer_gui_property():
    """Viewer exposes the chosen gui and threads it to the controller."""
    from cellier.convenience import Viewer

    viewer = Viewer(("z", "y", "x"), gui="anywidget")
    assert viewer.gui == "anywidget"
    assert viewer.controller._gui == "anywidget"


def test_ortho_viewer_gui_property():
    """OrthoViewer exposes the chosen gui and threads it to the controller."""
    from cellier.convenience import OrthoViewer

    viewer = OrthoViewer(("z", "y", "x"), gui="anywidget")
    assert viewer.gui == "anywidget"
    assert viewer.controller._gui == "anywidget"


def test_viewer_default_gui_is_qt():
    """Viewer defaults to the Qt gui."""
    from cellier.convenience import Viewer

    assert Viewer(("z", "y", "x")).gui == "qt"


# ---------------------------------------------------------------------------
# Task 6 - ControlPanel (dims) + make_dim_toggle
# ---------------------------------------------------------------------------


def _dims_changed_event(source_id, scene_id, *, displayed, slices, stacked=()):
    from cellier._state import AxisAlignedSelectionState, DimsState
    from cellier.events import DimsChangedEvent

    selection = AxisAlignedSelectionState(
        displayed_axes=displayed, slice_indices=slices, stacked_axes=stacked
    )
    state = DimsState(axis_labels=("z", "y", "x"), selection=selection)
    return DimsChangedEvent(
        source_id=source_id,
        scene_id=scene_id,
        dims_state=state,
        displayed_axes_changed=False,
    )


def _make_panel():
    from cellier.gui.anywidget import ControlPanel

    return ControlPanel()


def _make_dims_panel():
    from cellier.gui.anywidget import AnywidgetDimsPanel

    return AnywidgetDimsPanel(
        scene_id=uuid4(),
        axis_ranges={0: (0.0, 99.0), 1: (0.0, 511.0), 2: (0.0, 511.0)},
        axis_labels={0: "z", 1: "y", 2: "x"},
        slice_indices={0: 0},
        displayed_axes=(1, 2),
        stacked_axes=(),
    )


def test_panel_on_dims_changed_updates_without_emitting():
    """A model-driven dims change updates the trait but does not echo."""
    panel = _make_dims_panel()
    emitted = []
    panel.changed.connect(emitted.append)

    event = _dims_changed_event(
        source_id=uuid4(),
        scene_id=panel._scene_id,
        displayed=(1, 2),
        slices={0: 42},
    )
    panel._on_dims_changed(event)

    assert panel.slice_indices["0"] == 42
    assert emitted == []  # programmatic write is guarded, no re-emit


def test_panel_echo_filtered_by_source_id():
    """A DimsChangedEvent originating from the panel itself is ignored."""
    panel = _make_dims_panel()
    event = _dims_changed_event(
        source_id=panel._id,
        scene_id=panel._scene_id,
        displayed=(1, 2),
        slices={0: 7},
    )
    panel._on_dims_changed(event)
    assert "0" not in panel.slice_indices or panel.slice_indices["0"] == 0


def test_panel_user_change_emits_dims_update_event():
    """A user-driven trait change emits a DimsUpdateEvent for sliced axes only."""
    from cellier.events import DimsUpdateEvent

    panel = _make_dims_panel()
    emitted = []
    panel.changed.connect(emitted.append)

    # Simulate the JS save_changes write (string keys, all axes present).
    panel.slice_indices = {"0": 12, "1": 0, "2": 0}

    assert len(emitted) == 1
    event = emitted[0]
    assert isinstance(event, DimsUpdateEvent)
    assert event.source_id == panel._id
    assert event.scene_id == panel._scene_id
    # Only the sliced axis 0 is reported; displayed axes 1, 2 are excluded.
    assert event.slice_indices == {0: 12}


def test_make_dim_toggle_toggles_displayed_dimensions():
    """Clicking the toggle flips the viewer between 2D and 3D."""
    from cellier.convenience import Viewer
    from cellier.gui.anywidget import make_dim_toggle

    viewer = Viewer(("z", "y", "x"), dim="2d", gui="anywidget")
    toggle = make_dim_toggle(viewer)
    assert toggle.label == "Switch to 3D"
    assert len(viewer.scene.dims.selection.displayed_axes) == 2

    # Simulate a JS click (increment the synced counter).
    toggle._clicks += 1
    assert len(viewer.scene.dims.selection.displayed_axes) == 3
    assert toggle.label == "Switch to 2D"

    toggle._clicks += 1
    assert len(viewer.scene.dims.selection.displayed_axes) == 2
    assert toggle.label == "Switch to 3D"


# ---------------------------------------------------------------------------
# Task 7 - LayoutHost seam
# ---------------------------------------------------------------------------


def test_resolve_host_jupyter():
    """resolve_host('jupyter') returns a JupyterHost."""
    pytest.importorskip("IPython")
    from cellier.convenience._hosts import JupyterHost, resolve_host

    assert isinstance(resolve_host("jupyter"), JupyterHost)


def test_resolve_host_marimo():
    """resolve_host('marimo') returns a MarimoHost."""
    pytest.importorskip("marimo")
    from cellier.convenience._hosts import MarimoHost, resolve_host

    assert isinstance(resolve_host("marimo"), MarimoHost)


def test_resolve_host_unknown_raises():
    """An unknown explicit host that is not running raises."""
    from cellier.convenience._hosts import resolve_host

    with pytest.raises(RuntimeError, match="No anywidget host detected"):
        resolve_host("not-a-host")


def test_jupyter_host_stack_builds_awbox():
    """JupyterHost.stack composes its leaves into an AwBox."""
    from cellier.convenience._hosts import JupyterHost
    from cellier.gui.anywidget._container import AwBox

    host = JupyterHost()
    panel_a = _make_panel()
    panel_b = _make_panel()
    box = host.stack([host.leaf(panel_a), host.leaf(panel_b)], direction="v")
    assert isinstance(box, AwBox)
    assert box.direction == "v"
    assert list(box.children) == [panel_a, panel_b]


def test_jupyter_host_grid_builds_nested_awbox():
    """JupyterHost.grid nests horizontal AwBoxes inside a vertical AwBox."""
    from cellier.convenience._hosts import JupyterHost
    from cellier.gui.anywidget._container import AwBox

    host = JupyterHost()
    a, b, c, d = (_make_panel() for _ in range(4))
    grid = host.grid([[a, b], [c, d]])
    assert isinstance(grid, AwBox)
    assert grid.direction == "v"
    assert all(isinstance(row, AwBox) and row.direction == "h" for row in grid.children)


# ---------------------------------------------------------------------------
# Task 8 - builders return anywidget leaves with compose(host)
# ---------------------------------------------------------------------------


def _image_viewer():
    """A small anywidget Viewer with one in-memory image and a data store."""
    import numpy as np

    from cellier.convenience import Viewer, axis_ranges_from_viewer
    from cellier.data.image._image_memory_store import ImageMemoryStore

    blobs = np.zeros((8, 8, 8), dtype=np.float32)
    viewer = Viewer(("z", "y", "x"), dim="2d", gui="anywidget")
    store = ImageMemoryStore(data=blobs, name="blobs")
    viewer.controller.add_data_store(store)
    viewer.add_image(store, appearance={"color_map": "viridis", "clim": (0.0, 1.0)})
    return viewer, axis_ranges_from_viewer(viewer)


def _image_viewer_with_controls():
    """A small anywidget Viewer with one in-memory image and an appearance panel."""
    import numpy as np

    from cellier.convenience import Viewer, axis_ranges_from_viewer
    from cellier.data.image._image_memory_store import ImageMemoryStore

    blobs = np.zeros((8, 8, 8), dtype=np.float32)
    viewer = Viewer(("z", "y", "x"), dim="2d", gui="anywidget")
    store = ImageMemoryStore(data=blobs, name="blobs")
    viewer.controller.add_data_store(store)
    viewer.add_image(
        store,
        appearance={"color_map": "viridis", "clim": (0.0, 1.0)},
        controls={"appearance": ["color_map", "clim"]},
    )
    return viewer, axis_ranges_from_viewer(viewer)


def test_build_canvas_widget_anywidget_returns_view():
    """build_canvas_widget returns an AnywidgetCanvasView; no controls by default."""
    from cellier.convenience._hosts import JupyterHost
    from cellier.convenience.gui import AnywidgetCanvasView, build_canvas_widget

    viewer, ranges = _image_viewer()
    view = build_canvas_widget(viewer, ranges, gui="anywidget")

    assert isinstance(view, AnywidgetCanvasView)
    assert isinstance(view.canvas, rendercanvas_anywidget.RenderCanvas)
    # No controls config was registered, so the panel is absent.
    assert view.controls is None
    assert callable(view.compose)

    # Without controls, compose() returns only the right column (no h-stack).
    composed = view.compose(JupyterHost())
    from cellier.gui.anywidget._container import AwBox

    assert isinstance(composed, AwBox)
    assert composed.direction == "v"


def test_build_canvas_widget_anywidget_with_controls():
    """build_canvas_widget always returns view.controls=None; the renderer builds it."""
    from cellier.convenience._hosts import JupyterHost
    from cellier.convenience.gui import AnywidgetCanvasView, build_canvas_widget

    viewer, ranges = _image_viewer_with_controls()
    view = build_canvas_widget(viewer, ranges, gui="anywidget")

    assert isinstance(view, AnywidgetCanvasView)
    # Panel building is deferred to the renderer; the canvas view never holds it.
    assert view.controls is None

    composed = view.compose(JupyterHost())
    from cellier.gui.anywidget._container import AwBox

    assert isinstance(composed, AwBox)
    # No controls in the canvas view, so compose returns only the right column.
    assert composed.direction == "v"


def test_build_canvas_widget_gui_conflict_raises():
    """A gui that conflicts with viewer.gui is rejected."""
    from cellier.convenience.gui import build_canvas_widget

    viewer, ranges = _image_viewer()
    with pytest.raises(ValueError, match="conflicts with viewer"):
        build_canvas_widget(viewer, ranges, gui="qt")


def test_build_canvas_widget_defaults_gui_from_viewer():
    """Omitting gui defaults to viewer.gui."""
    from cellier.convenience.gui import AnywidgetCanvasView, build_canvas_widget

    viewer, ranges = _image_viewer()
    view = build_canvas_widget(viewer, ranges)
    assert isinstance(view, AnywidgetCanvasView)


def test_build_ortho_grid_anywidget_returns_canvases():
    """build_ortho_grid_widget(gui='anywidget') returns a composable grid."""
    import numpy as np

    from cellier.convenience import OrthoViewer, axis_ranges_from_ortho
    from cellier.convenience._hosts import JupyterHost
    from cellier.convenience.gui import OrthoAnywidgetCanvases, build_ortho_grid_widget
    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.gui.anywidget._container import AwBox

    viewer = OrthoViewer(("z", "y", "x"), gui="anywidget")
    store = ImageMemoryStore(data=np.zeros((8, 8, 8), dtype=np.float32), name="blobs")
    viewer.controller.add_data_store(store)
    viewer.add_image(store, appearance={"color_map": "viridis", "clim": (0.0, 1.0)})
    ranges = axis_ranges_from_ortho(viewer)

    grid = build_ortho_grid_widget(viewer, ranges, gui="anywidget")
    assert isinstance(grid, OrthoAnywidgetCanvases)
    assert set(grid.canvases) == {"xy", "xz", "yz", "vol"}

    composed = grid.compose(JupyterHost())
    assert isinstance(composed, AwBox)


# ---------------------------------------------------------------------------
# Task 9 - display() launcher
# ---------------------------------------------------------------------------


class _Node:
    """A composable node that can hold an attached close() (unlike a tuple)."""

    def __init__(self, kind, payload):
        self.kind = kind
        self.payload = payload


class _FakeHost:
    """An imperative LayoutHost (Jupyter-like): renders in present, returns None."""

    def __init__(self):
        self.presented = None

    def leaf(self, widget):
        return _Node("leaf", widget)

    def stack(self, items, *, direction="v", align=None, min_width=None):
        return _Node("stack", (direction, list(items), align, min_width))

    def grid(self, rows):
        return _Node("grid", [list(r) for r in rows])

    def present(self, root):
        self.presented = root
        return None  # imperative: rendered as a side effect


class _FakeReturnValueHost(_FakeHost):
    """A return-value LayoutHost (marimo-like): present returns the renderable."""

    def present(self, root):
        self.presented = root
        return root


def test_display_imports_from_convenience():
    """display, Layout, AppearanceControls, SceneControls are importable."""
    from cellier.convenience import AppearanceControls, Layout, SceneControls, display

    assert callable(display)
    assert Layout is not None
    assert AppearanceControls is not None
    assert SceneControls is not None


def test_display_center_only_presents_and_returns_inert_handle(monkeypatch):
    """Layout with no docks: presents just the center, returns an inert handle."""
    from cellier.convenience import _hosts, _launch
    from cellier.convenience._launch import DisplayHandle
    from cellier.convenience.gui import build_canvas_widget
    from cellier.convenience.layout import Layout

    viewer, ranges = _image_viewer()
    view = build_canvas_widget(viewer, ranges, gui="anywidget")

    fake = _FakeHost()
    monkeypatch.setattr(_hosts, "resolve_host", lambda host=None: fake)

    handle = _launch.display(viewer, Layout(center=view), fit="none")

    # present() was called exactly once.
    assert fake.presented is not None
    # No docks: root is just the canvas+dims v-stack from compose().
    assert fake.presented.kind == "stack"
    _direction, leaves, _align, _min_width = fake.presented.payload
    assert _direction == "v"
    assert len(leaves) == 2  # canvas leaf + dims leaf

    # display() returns an inert handle (not the renderable) for Jupyter.
    assert isinstance(handle, DisplayHandle)
    assert handle._repr_mimebundle_() == {}

    # close() is idempotent.
    handle.close()
    handle.close()


def test_display_bottom_dock_stacks_toggle_below_center(monkeypatch):
    """SceneControls in bottom_dock produces a v-stack: [center, toggle]."""
    from cellier.convenience import _hosts, _launch
    from cellier.convenience.gui import build_canvas_widget
    from cellier.convenience.layout import Layout, SceneControls

    viewer, ranges = _image_viewer()
    view = build_canvas_widget(viewer, ranges, gui="anywidget")

    fake = _FakeHost()
    monkeypatch.setattr(_hosts, "resolve_host", lambda host=None: fake)

    _launch.display(
        viewer, Layout(center=view, bottom_dock=SceneControls()), fit="none"
    )

    # Outer v-stack: [center_composed, toggle_leaf]
    presented = fake.presented
    assert presented.kind == "stack"
    direction, leaves, align, _min_width = presented.payload
    assert direction == "v"
    # No align: default cross-axis "stretch" is required for the responsive-
    # width layout to fill the notebook cell / sidecar tab (see
    # _anywidget_renderer.py); "center" would shrink-to-content again.
    assert align is None
    assert len(leaves) == 2
    center_node, toggle_node = leaves
    assert center_node.kind == "stack"  # canvas+dims v-stack
    assert toggle_node.kind == "leaf"  # SceneControls -> host.leaf(toggle)


def test_display_left_dock_stacks_controls_beside_center(monkeypatch):
    """AppearanceControls in left_dock produces a h-stack: [panel, center]."""
    from cellier.convenience import _hosts, _launch
    from cellier.convenience.gui import build_canvas_widget
    from cellier.convenience.layout import AppearanceControls, Layout

    viewer, ranges = _image_viewer_with_controls()
    view = build_canvas_widget(viewer, ranges, gui="anywidget")

    fake = _FakeHost()
    monkeypatch.setattr(_hosts, "resolve_host", lambda host=None: fake)

    _launch.display(
        viewer, Layout(center=view, left_dock=AppearanceControls()), fit="none"
    )

    # Middle h-stack: [panel_leaf, center_composed]
    presented = fake.presented
    assert presented.kind == "stack"
    direction, leaves, _align, _min_width = presented.payload
    assert direction == "h"
    assert len(leaves) == 2
    panel_node, center_node = leaves
    assert panel_node.kind == "leaf"  # AppearanceControls -> host.leaf(panel)
    assert center_node.kind == "stack"  # canvas+dims v-stack


def test_display_return_value_host_yields_renderable(monkeypatch):
    """A return-value host (marimo-like) gets the renderable back from display."""
    from cellier.convenience import _hosts, _launch
    from cellier.convenience.gui import build_canvas_widget
    from cellier.convenience.layout import Layout, SceneControls

    viewer, ranges = _image_viewer()
    view = build_canvas_widget(viewer, ranges, gui="anywidget")

    fake = _FakeReturnValueHost()
    monkeypatch.setattr(_hosts, "resolve_host", lambda host=None: fake)

    result = _launch.display(
        viewer, Layout(center=view, bottom_dock=SceneControls()), fit="none"
    )

    # Return-value host: display() yields the renderable so marimo renders it.
    assert result is fake.presented
    # teardown is attached best-effort to the renderable.
    assert callable(result.close)
    result.close()


def test_run_dispatches_to_display_for_anywidget(monkeypatch):
    """run() with gui='anywidget' calls display() and returns a DisplayHandle."""
    from cellier.convenience import _hosts, _launch
    from cellier.convenience._launch import DisplayHandle
    from cellier.convenience.gui import build_canvas_widget
    from cellier.convenience.layout import Layout

    viewer, ranges = _image_viewer()
    view = build_canvas_widget(viewer, ranges, gui="anywidget")

    fake = _FakeHost()
    monkeypatch.setattr(_hosts, "resolve_host", lambda host=None: fake)

    result = _launch.run(viewer, Layout(center=view), fit="none")

    assert fake.presented is not None
    assert isinstance(result, DisplayHandle)


def test_run_importable_from_convenience():
    """run is importable from cellier.convenience."""
    from cellier.convenience import run

    assert callable(run)


def test_layout_single_preset_no_docks():
    """Layout.single with no dock args produces a center-only layout."""
    from cellier.convenience.layout import Layout

    viewer, ranges = _image_viewer()
    from cellier.convenience.gui import build_canvas_widget

    view = build_canvas_widget(viewer, ranges, gui="anywidget")
    layout = Layout.single(view)
    assert layout.center is view
    assert layout.left_dock is None
    assert layout.right_dock is None
    assert layout.top_dock is None
    assert layout.bottom_dock is None


def test_layout_single_preset_with_docks():
    """Layout.single wires appearance and scene_controls to the requested docks."""
    from cellier.convenience.layout import AppearanceControls, Layout, SceneControls

    viewer, ranges = _image_viewer()
    from cellier.convenience.gui import build_canvas_widget

    view = build_canvas_widget(viewer, ranges, gui="anywidget")
    layout = Layout.single(view, appearance="left", scene_controls="bottom")
    assert isinstance(layout.left_dock, AppearanceControls)
    assert isinstance(layout.bottom_dock, SceneControls)
    assert layout.right_dock is None
    assert layout.top_dock is None


# ---------------------------------------------------------------------------
# Phase 2 - appearance controls
# ---------------------------------------------------------------------------


def _make_panel_with_visual():
    """Return a ControlPanel wired to a fake visual via appearance_fields."""
    from cellier.gui.anywidget import ControlPanel

    return ControlPanel(
        visual_id=uuid4(),
        appearance_fields=[
            "color_map",
            "clim",
            "render_mode",
            "iso_threshold",
            "attenuation",
            "lod_bias",
        ],
        render_mode="mip",
        iso_threshold=0.2,
        attenuation=1.0,
        lod_bias=1.0,
        clim=[0.0, 1.0],
        clim_range=[0.0, 1.0],
        color_map="grays",
    )


def test_panel_has_appearance_trait_when_visual_id_given():
    """has_appearance=True and appearance_fields non-empty when visual_id is set."""
    panel = _make_panel_with_visual()
    assert panel.has_appearance is True
    assert "render_mode" in panel.appearance_fields
    assert "clim" in panel.appearance_fields
    assert "color_map" in panel.appearance_fields


def test_panel_has_no_appearance_when_no_visual():
    """has_appearance=False when no visual_id is supplied."""
    panel = _make_panel()
    assert panel.has_appearance is False
    assert panel.appearance_fields == []


def _appearance_changed_event(source_id, visual_id, field, value):
    from cellier.events import AppearanceChangedEvent

    return AppearanceChangedEvent(
        source_id=source_id,
        visual_id=visual_id,
        field_name=field,
        new_value=value,
        requires_reslice=False,
    )


def test_panel_on_appearance_changed_updates_trait_without_emitting():
    """A model-driven AppearanceChangedEvent updates the trait but does not echo."""
    panel = _make_panel_with_visual()
    emitted = []
    panel.changed.connect(emitted.append)

    event = _appearance_changed_event(
        source_id=uuid4(),
        visual_id=panel._visual_id,
        field="render_mode",
        value="iso",
    )
    panel._on_appearance_changed(event)

    assert panel.render_mode == "iso"
    assert emitted == []


def test_panel_appearance_echo_filtered_by_source_id():
    """An AppearanceChangedEvent from the panel itself is ignored."""
    panel = _make_panel_with_visual()
    original = panel.render_mode

    event = _appearance_changed_event(
        source_id=panel._id,  # echo: same source
        visual_id=panel._visual_id,
        field="render_mode",
        value="iso",
    )
    panel._on_appearance_changed(event)
    assert panel.render_mode == original


def test_panel_appearance_changed_colormap_converts_to_str():
    """color_map in AppearanceChangedEvent is converted to string name."""
    from cmap import Colormap

    panel = _make_panel_with_visual()
    event = _appearance_changed_event(
        source_id=uuid4(),
        visual_id=panel._visual_id,
        field="color_map",
        value=Colormap("viridis"),
    )
    panel._on_appearance_changed(event)
    # Should be stored as a string, not a Colormap object
    assert isinstance(panel.color_map, str)
    assert "viridis" in panel.color_map


def test_panel_user_render_mode_change_emits_appearance_update():
    """A user-driven render_mode change emits AppearanceUpdateEvent."""
    from cellier.events import AppearanceUpdateEvent

    panel = _make_panel_with_visual()
    emitted = []
    panel.changed.connect(emitted.append)

    panel.render_mode = "iso"

    assert len(emitted) == 1
    event = emitted[0]
    assert isinstance(event, AppearanceUpdateEvent)
    assert event.source_id == panel._id
    assert event.visual_id == panel._visual_id
    assert event.field == "render_mode"
    assert event.value == "iso"


def test_panel_user_clim_change_emits_appearance_update():
    """A user-driven clim change emits AppearanceUpdateEvent with a tuple value."""
    from cellier.events import AppearanceUpdateEvent

    panel = _make_panel_with_visual()
    emitted = []
    panel.changed.connect(emitted.append)

    panel.clim = [0.1, 0.9]

    assert len(emitted) == 1
    event = emitted[0]
    assert isinstance(event, AppearanceUpdateEvent)
    assert event.field == "clim"
    assert event.value == (0.1, 0.9)  # converted list -> tuple


def test_panel_user_lod_bias_change_emits_appearance_update():
    """A user-driven lod_bias change emits AppearanceUpdateEvent."""
    from cellier.events import AppearanceUpdateEvent

    panel = _make_panel_with_visual()
    emitted = []
    panel.changed.connect(emitted.append)

    panel.lod_bias = 2.0

    assert len(emitted) == 1
    event = emitted[0]
    assert isinstance(event, AppearanceUpdateEvent)
    assert event.field == "lod_bias"
    assert event.value == 2.0


def _aabb_changed_event(source_id, visual_id, field, value):
    from cellier.events import AABBChangedEvent

    return AABBChangedEvent(
        source_id=source_id,
        visual_id=visual_id,
        field_name=field,
        new_value=value,
    )


def test_panel_on_aabb_changed_updates_trait_without_emitting():
    """A model-driven AABBChangedEvent updates the trait but does not echo."""
    panel = _make_panel_with_visual()
    emitted = []
    panel.changed.connect(emitted.append)

    event = _aabb_changed_event(
        source_id=uuid4(),
        visual_id=panel._visual_id,
        field="enabled",
        value=True,
    )
    panel._on_aabb_changed(event)

    assert panel.aabb_enabled is True
    assert emitted == []


def test_panel_user_aabb_enabled_change_emits_aabb_update():
    """A user-driven aabb_enabled change emits AABBUpdateEvent."""
    from cellier.events import AABBUpdateEvent

    panel = _make_panel_with_visual()
    emitted = []
    panel.changed.connect(emitted.append)

    panel.aabb_enabled = True

    assert len(emitted) == 1
    event = emitted[0]
    assert isinstance(event, AABBUpdateEvent)
    assert event.source_id == panel._id
    assert event.visual_id == panel._visual_id
    assert event.field == "enabled"
    assert event.value is True


def test_panel_user_aabb_color_change_emits_aabb_update():
    """A user-driven aabb_color change emits AABBUpdateEvent with field 'color'."""
    from cellier.events import AABBUpdateEvent

    panel = _make_panel_with_visual()
    emitted = []
    panel.changed.connect(emitted.append)

    panel.aabb_color = "#ff0000"

    assert len(emitted) == 1
    event = emitted[0]
    assert isinstance(event, AABBUpdateEvent)
    assert event.field == "color"
    assert event.value == "#ff0000"


def test_panel_appearance_only_emits_for_declared_fields():
    """Changes to undeclared fields are not emitted."""
    from cellier.gui.anywidget import ControlPanel

    panel = ControlPanel(
        visual_id=uuid4(),
        appearance_fields=["render_mode"],  # only render_mode declared
        render_mode="mip",
        lod_bias=1.0,
    )
    emitted = []
    panel.changed.connect(emitted.append)

    # lod_bias is not in appearance_fields so no event should be emitted
    panel.lod_bias = 2.5
    assert emitted == []


def test_from_scene_with_visual_populates_appearance():
    """from_scene(visual=...) enables appearance controls with the visual's fields."""
    import numpy as np

    from cellier.convenience import Viewer
    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.gui.anywidget import ControlPanel

    viewer = Viewer(("z", "y", "x"), gui="anywidget")
    store = ImageMemoryStore(data=np.zeros((4, 4, 4), dtype=np.float32), name="t")
    viewer.controller.add_data_store(store)
    viewer.add_image(store, appearance={"color_map": "viridis", "clim": (0.0, 1.0)})

    scene = viewer.scene
    visual = scene.visuals[0]
    axis_ranges = {0: (0.0, 3.0), 1: (0.0, 3.0), 2: (0.0, 3.0)}

    panel = ControlPanel.from_scene(scene, axis_ranges, visual=visual)

    assert isinstance(panel, ControlPanel)
    assert panel.has_appearance is True
    assert panel._visual_id == visual.id
    assert "color_map" in panel.appearance_fields
    assert "clim" in panel.appearance_fields
    assert "viridis" in panel.color_map


def test_renderer_builds_appearance_panel_for_configured_visual(monkeypatch):
    """The anywidget renderer creates an AppearanceControls panel in the left dock."""
    import numpy as np

    from cellier.convenience import Viewer, _hosts, _launch, axis_ranges_from_viewer
    from cellier.convenience.gui import build_canvas_widget
    from cellier.convenience.layout import AppearanceControls, Layout
    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.events import AABBChangedEvent, AppearanceChangedEvent
    from cellier.gui.anywidget import ControlPanel

    viewer = Viewer(("z", "y", "x"), gui="anywidget")
    store = ImageMemoryStore(data=np.zeros((4, 4, 4), dtype=np.float32), name="t")
    viewer.controller.add_data_store(store)
    viewer.add_image(
        store,
        appearance={"color_map": "viridis", "clim": (0.0, 1.0)},
        controls={"appearance": ["color_map", "clim"]},
    )

    ranges = axis_ranges_from_viewer(viewer)
    view = build_canvas_widget(viewer, ranges, gui="anywidget")
    # build_canvas_widget no longer builds the panel.
    assert view.controls is None

    fake = _FakeHost()
    monkeypatch.setattr(_hosts, "resolve_host", lambda host=None: fake)

    _launch.display(
        viewer, Layout(center=view, left_dock=AppearanceControls()), fit="none"
    )

    # The renderer placed the panel as the left leaf.
    presented = fake.presented
    _direction, leaves, _align, _min_width = presented.payload
    panel_leaf = leaves[0]
    assert panel_leaf.kind == "leaf"
    panel = panel_leaf.payload
    assert isinstance(panel, ControlPanel)
    assert panel.has_appearance is True
    assert panel._visual_id is not None
    event_types = {s.event_type for s in panel.subscription_specs()}
    assert AppearanceChangedEvent in event_types
    assert AABBChangedEvent in event_types
