"""The two layout renderers produce the same thing from the same spec.

``test_render_controls_parity.py`` pins that both front ends draw the same
*controls*; this pins that they compose them the same way.  Every assertion
here failed when it was written -- see ``plans/gui_backend_unification.md``
section 2.1 -- so each one names a real difference between what a Qt user and
a notebook user saw from identical code.

The two renderers cannot share a construction path (one builds ``QWidget``s,
the other serialises to traits), so nothing but a test can hold them
together.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.data._dataset_info import DatasetInfo, MatrixSection, RowSection

_MESH_POSITIONS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
)
_MESH_INDICES = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
_RANGES = {0: (0.0, 2.0), 1: (0.0, 2.0), 2: (0.0, 2.0)}


def _mesh_store():
    from cellier.data.mesh._mesh_memory_store import MeshMemoryStore

    return MeshMemoryStore(positions=_MESH_POSITIONS, indices=_MESH_INDICES)


def _viewer(gui: str, *, configure_controls: bool = False):
    """A one-mesh viewer, with or without a recorded ``controls=`` config."""
    from cellier.convenience import MeshControlsConfig, Viewer
    from cellier.visuals._mesh_memory import MeshFlatAppearance

    viewer = Viewer(("z", "y", "x"), dim="3d", gui=gui)
    viewer.add_mesh(
        _mesh_store(),
        appearance=MeshFlatAppearance(),
        controls=MeshControlsConfig(appearance=True) if configure_controls else None,
    )
    return viewer


# ---------------------------------------------------------------------------
# D1: closing a window releases what the renderer built -- all of it
# ---------------------------------------------------------------------------


def test_closing_a_qt_window_closes_the_center_too(qtbot):
    """The dock is not the only thing the renderer built.

    ``render_qt`` threaded its ``closeables`` list into the docks and not into
    the center, so ``QtCanvasWidget.close()`` was never called and the dims
    control stayed subscribed to the bus for the life of the controller.  The
    anywidget renderer closes its center leaves, which is why only Qt leaked.

    Asserted on the dims control specifically rather than on a subscription
    count, because a count shrinking says only that *something* was released.
    """
    pytest.importorskip("qtpy")
    pytest.importorskip("superqt")
    from cellier.convenience import AppearanceControls, Layout
    from cellier.convenience.gui import build_canvas_widget
    from cellier.convenience.layout._qt_renderer import render_qt

    viewer = _viewer("qt", configure_controls=True)
    canvas = build_canvas_widget(viewer, _RANGES)
    window = render_qt(Layout(center=canvas, left_dock=AppearanceControls()), viewer)

    dims_id = canvas.dims_control._id
    assert dims_id in _subscribed_owners(viewer.controller), "sanity: it subscribed"

    window.close()

    assert dims_id not in _subscribed_owners(viewer.controller)


def _subscribed_owners(controller) -> set:
    return {
        getattr(subscription, "owner_id", None)
        for subscriptions in controller._outgoing_events._subs.values()
        for subscription in subscriptions
    }


# ---------------------------------------------------------------------------
# D2: a Grid cell holds its column on both toolkits
# ---------------------------------------------------------------------------


class _FakeLeaf:
    """A center leaf both renderers accept: ``.widget`` for Qt, ``.compose``."""

    def __init__(self, name: str) -> None:
        self.name = name

    def compose(self, host):  # anywidget
        return self.name

    @property
    def widget(self):  # Qt
        from qtpy.QtWidgets import QLabel

        self._label = QLabel(self.name)
        return self._label


class _RecordingHost:
    """A ``LayoutHost`` that records the tree instead of building widgets."""

    def leaf(self, widget):
        return widget

    def stack(self, items, *, direction="v", **kwargs):
        return (f"stack-{direction}", list(items))

    def grid(self, rows):
        return ("grid", [list(row) for row in rows])

    def present(self, root):
        return root


def test_an_empty_grid_cell_does_not_shift_the_next_one_left():
    """``None`` leaves a cell empty; it does not close the gap.

    The anywidget renderer filtered ``None`` out of each row, so
    ``[[a, None], [None, b]]`` put ``b`` in column 0 where Qt put it in
    column 1 -- the same spec, two different pictures.  ``Grid.cells``
    documents the Qt behaviour, so anywidget was the side that was wrong.
    """
    from cellier.convenience.layout._anywidget_renderer import _render_center
    from cellier.convenience.layout._spec import Grid

    grid = Grid(cells=[[_FakeLeaf("a"), None], [None, _FakeLeaf("b")]])
    kind, rows = _render_center(grid, _RecordingHost(), [])

    assert kind == "grid"
    assert rows == [["a", None], [None, "b"]]


def test_the_qt_grid_puts_the_same_cell_in_the_same_column(qtbot):
    """The reference behaviour the test above is asserted against."""
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    from cellier.convenience.layout._qt_renderer import _render_center_qt
    from cellier.convenience.layout._spec import Grid

    QApplication.instance() or QApplication([])
    leaf_a, leaf_b = _FakeLeaf("a"), _FakeLeaf("b")
    container = _render_center_qt(Grid(cells=[[leaf_a, None], [None, leaf_b]]))

    layout = container.layout()
    positions = {
        layout.itemAt(index).widget().text(): layout.getItemPosition(index)[:2]
        for index in range(layout.count())
    }
    assert positions == {"a": (0, 0), "b": (1, 1)}


# ---------------------------------------------------------------------------
# D3: a dock whose contents all resolve to nothing produces no dock
# ---------------------------------------------------------------------------


def test_a_dock_stack_that_builds_nothing_produces_no_dock(qtbot):
    """``AppearanceControls`` on an unconfigured viewer builds nothing.

    Bare, both renderers already agreed -- the spec resolves to ``None`` and
    the dock is skipped.  Wrapped in a stack they did not: the Qt branch
    returned its container unconditionally, so an empty ``QDockWidget``
    titled "Left" appeared beside the canvas.
    """
    pytest.importorskip("qtpy")
    pytest.importorskip("superqt")
    from qtpy.QtWidgets import QDockWidget

    from cellier.convenience import AppearanceControls, Layout, VStack
    from cellier.convenience.gui import build_canvas_widget
    from cellier.convenience.layout._anywidget_renderer import _render_dock
    from cellier.convenience.layout._qt_renderer import render_qt

    spec = VStack(items=[AppearanceControls()])

    qt_viewer = _viewer("qt")  # no controls= -> nothing to build
    window = render_qt(
        Layout(center=build_canvas_widget(qt_viewer, _RANGES), left_dock=spec),
        qt_viewer,
    )
    assert window.findChildren(QDockWidget) == []

    anywidget_viewer = _viewer("anywidget")
    assert _render_dock(spec, anywidget_viewer, _RecordingHost(), []) is None


# ---------------------------------------------------------------------------
# D4: dataset-info sections are drawn in the order the store declared them
# ---------------------------------------------------------------------------


def _sample_dataset_info() -> DatasetInfo:
    """The section shape ``dataset_info_from_path`` actually builds.

    A matrix between two labelled row sections is the case that separated the
    two toolkits, and it is what every multiscale OME-Zarr store produces.
    """
    return DatasetInfo(
        sections=[
            RowSection(None, [("Store", "zarr")]),
            RowSection("Axes", [("z", "um")]),
            MatrixSection(
                "World to data",
                np.eye(2),
                row_labels=["z", "1"],
                col_labels=["z", "1"],
            ),
            RowSection("Scale levels", [("0", "10x10")], collapsed=True),
        ]
    )


_EXPECTED_SECTION_ORDER = ["Store", "Axes", "World to data", "Scale levels"]


def test_both_front_ends_draw_the_sections_in_declaration_order(qtbot):
    """Qt hoisted matrices above labelled row sections.

    Unlabelled rows and every matrix went into a top-level form, while
    labelled row sections became nested collapsibles appended after it -- so
    a store's declared order survived only when it happened to match that
    split.  For the pyramid above it did not: the transform matrix jumped
    over "Axes", and every multiscale OME-Zarr panel read differently
    depending on the toolkit.
    """
    pytest.importorskip("qtpy")
    pytest.importorskip("superqt")
    from cellier.gui.anywidget import AnywidgetDatasetInfo
    from cellier.gui.qt import QtDatasetInfo

    info = _sample_dataset_info()
    qt_labels = QtDatasetInfo.from_info(info).section_labels()
    anywidget_labels = AnywidgetDatasetInfo.from_info(info).section_labels()

    assert qt_labels == anywidget_labels == _EXPECTED_SECTION_ORDER


# ---------------------------------------------------------------------------
# D5: the render dock names its scope on both toolkits
# ---------------------------------------------------------------------------


def test_both_render_docks_carry_the_dock_title(qtbot):
    """ "Outline" beside "Outlines" is not a distinction anyone should notice.

    The Qt dock wraps its panels in a group box titled "Renderer effects" so
    the global settings cannot be mistaken for the per-visual ones docked on
    the other side of the canvas.  The notebook dock stacked its panels bare,
    so the notebook had exactly the confusion the heading exists to prevent.
    """
    pytest.importorskip("qtpy")
    pytest.importorskip("superqt")
    from cellier.convenience.layout._anywidget_renderer import _render_dock
    from cellier.convenience.layout._qt_renderer import _render_dock_qt
    from cellier.convenience.layout._spec import RenderControls
    from cellier.gui._render_controls import RENDER_DOCK_TITLE

    spec = RenderControls(sections=("outline",))

    qt_dock = _render_dock_qt(spec, _viewer("qt"), [])
    assert qt_dock.title() == RENDER_DOCK_TITLE

    anywidget_dock = _render_dock(spec, _viewer("anywidget"), _TitleRecordingHost(), [])
    assert anywidget_dock[0] == RENDER_DOCK_TITLE


class _TitleRecordingHost(_RecordingHost):
    """Records the title a stack was given, which is what D5 is about."""

    def stack(self, items, *, direction="v", title=None, **kwargs):
        return (title, list(items))


# ---------------------------------------------------------------------------
# D11: both front ends coalesce slider drags at the same rate
# ---------------------------------------------------------------------------


def test_both_dims_panels_throttle_at_the_same_interval(qtbot):
    """The interval is what a user feels, so it cannot be declared twice.

    Qt throttles with a ``QTimer`` and anywidget with a ``setTimeout`` loop in
    ``dims_panel.js`` -- two implementations, which is fine, over what used to
    be two independent 50s, one in Python and one in JavaScript where no test
    could reach it.  The JS now reads a synced trait instead.
    """
    pytest.importorskip("qtpy")
    pytest.importorskip("superqt")
    from cellier.gui._constants import DIMS_SLIDER_THROTTLE_MS
    from cellier.gui.anywidget._dims_panel import AnywidgetDimsPanel
    from cellier.gui.qt._scene import QtDimsControl

    qt_dims = QtDimsControl(
        scene_id=uuid4(),
        axis_ranges={0: (0.0, 4.0)},
        axis_labels={0: "z"},
        initial_slice_indices={0: 0},
    )
    qtbot.addWidget(qt_dims.widget)
    anywidget_dims = AnywidgetDimsPanel(
        scene_id=uuid4(),
        axis_ranges={0: (0.0, 4.0)},
        axis_labels={0: "z"},
        slice_indices={0: 0},
    )

    assert qt_dims._rate_limit_timer.interval() == DIMS_SLIDER_THROTTLE_MS
    assert anywidget_dims.throttle_ms == DIMS_SLIDER_THROTTLE_MS


def test_the_dims_javascript_reads_the_interval_rather_than_declaring_one():
    """A hard-coded constant in the ESM is unreachable from any Python test."""
    from cellier.gui.anywidget import _dims_panel

    source = (_dims_panel._STATIC / "dims_panel.js").read_text()
    assert 'model.get("throttle_ms")' in source
    assert "THROTTLE_MS = 50" not in source
