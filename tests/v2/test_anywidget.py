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

    return ControlPanel(
        scene_id=uuid4(),
        axis_ranges={0: (0.0, 99.0), 1: (0.0, 511.0), 2: (0.0, 511.0)},
        axis_labels={0: "z", 1: "y", 2: "x"},
        slice_indices={0: 0},
        displayed_axes=(1, 2),
        stacked_axes=(),
    )


def test_panel_on_dims_changed_updates_without_emitting():
    """A model-driven dims change updates the trait but does not echo."""
    panel = _make_panel()
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
    panel = _make_panel()
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

    panel = _make_panel()
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


def test_build_canvas_widget_anywidget_returns_view():
    """build_canvas_widget(gui='anywidget') returns a composable view."""
    from cellier.convenience._hosts import JupyterHost
    from cellier.convenience.gui import AnywidgetCanvasView, build_canvas_widget
    from cellier.gui.anywidget import ControlPanel

    viewer, ranges = _image_viewer()
    view = build_canvas_widget(viewer, ranges, gui="anywidget")

    assert isinstance(view, AnywidgetCanvasView)
    assert isinstance(view.canvas, rendercanvas_anywidget.RenderCanvas)
    assert isinstance(view.controls, ControlPanel)
    assert callable(view.compose)

    composed = view.compose(JupyterHost())
    from cellier.gui.anywidget._container import AwBox

    assert isinstance(composed, AwBox)
    # canvas-over-controls column is centred on the cross-axis.
    assert composed.align == "center"


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

    def stack(self, items, *, direction="v", align=None):
        return _Node("stack", (direction, list(items), align))

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
    """display and make_dim_toggle are importable from cellier.convenience."""
    from cellier.convenience import display, make_dim_toggle

    assert callable(display)
    assert callable(make_dim_toggle)


def test_display_composes_presents_and_returns_inert_handle(monkeypatch):
    """display composes controls above the view, presents once, returns a handle."""
    from cellier.convenience import _hosts, _launch
    from cellier.convenience._launch import DisplayHandle
    from cellier.convenience.gui import build_canvas_widget
    from cellier.gui.anywidget import make_dim_toggle

    viewer, ranges = _image_viewer()
    view = build_canvas_widget(viewer, ranges, gui="anywidget")
    toggle = make_dim_toggle(viewer)

    fake = _FakeHost()
    monkeypatch.setattr(_hosts, "resolve_host", lambda host=None: fake)

    handle = _launch.display(viewer, view, controls=[toggle], fit="none")

    # present() rendered the composed stack exactly once.
    assert fake.presented is not None
    assert fake.presented.kind == "stack"
    _direction, leaves, align = fake.presented.payload
    # the outer column is centred so controls line up under the viewer.
    assert align == "center"
    # control leaf comes first; the view's composed node is last.
    assert leaves[0].kind == "leaf" and leaves[0].payload is toggle
    assert leaves[-1].kind == "stack"

    # display() returns an inert handle (not the renderable) so the cell shows
    # a single copy; the handle has no representation of its own.
    assert isinstance(handle, DisplayHandle)
    assert handle._repr_mimebundle_() == {}

    # close() tears down idempotently.
    handle.close()
    handle.close()


def test_display_return_value_host_yields_renderable(monkeypatch):
    """A return-value host (marimo-like) gets the renderable back from display."""
    from cellier.convenience import _hosts, _launch
    from cellier.convenience.gui import build_canvas_widget
    from cellier.gui.anywidget import make_dim_toggle

    viewer, ranges = _image_viewer()
    view = build_canvas_widget(viewer, ranges, gui="anywidget")
    toggle = make_dim_toggle(viewer)

    fake = _FakeReturnValueHost()
    monkeypatch.setattr(_hosts, "resolve_host", lambda host=None: fake)

    result = _launch.display(viewer, view, controls=[toggle], fit="none")

    # display() returns the host renderable (so marimo's last expression renders
    # it), not the inert handle.
    assert result is fake.presented
    assert result.kind == "stack"
    # teardown is attached best-effort to the renderable.
    assert callable(result.close)
    result.close()


def test_display_controls_position_bottom(monkeypatch):
    """controls_position='bottom' stacks the controls below the view."""
    from cellier.convenience import _hosts, _launch
    from cellier.convenience.gui import build_canvas_widget
    from cellier.gui.anywidget import make_dim_toggle

    viewer, ranges = _image_viewer()
    view = build_canvas_widget(viewer, ranges, gui="anywidget")
    toggle = make_dim_toggle(viewer)

    fake = _FakeHost()
    monkeypatch.setattr(_hosts, "resolve_host", lambda host=None: fake)

    _launch.display(
        viewer, view, controls=[toggle], controls_position="bottom", fit="none"
    )

    _direction, leaves, _align = fake.presented.payload
    # view first, control leaf last.
    assert leaves[0].kind == "stack"
    assert leaves[-1].kind == "leaf" and leaves[-1].payload is toggle


def test_display_controls_position_invalid(monkeypatch):
    """An unknown controls_position is rejected."""
    from cellier.convenience import _hosts, _launch
    from cellier.convenience.gui import build_canvas_widget

    viewer, ranges = _image_viewer()
    view = build_canvas_widget(viewer, ranges, gui="anywidget")
    monkeypatch.setattr(_hosts, "resolve_host", lambda host=None: _FakeHost())

    with pytest.raises(ValueError, match="controls_position"):
        _launch.display(viewer, view, controls_position="left")
