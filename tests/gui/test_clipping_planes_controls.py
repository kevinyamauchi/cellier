"""The "Clipping planes" control on both toolkits (clipping planes design 7).

The control sends ``ClippingPlanesUpdateEvent`` and follows
``ClippingPlanesChangedEvent``.
"""

from __future__ import annotations

import itertools
from uuid import uuid4

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data import PointsMemoryStore
from cellier.gui._clipping_planes import (
    CLIPPING_PLANES_TITLE,
    clipping_planes_seed,
    facing_of,
    planes_from_rows,
    position_range,
    rows_from_planes,
)
from cellier.transform import Axis, DataCoordinateSystem
from cellier.visuals import ClippingPlane

_SERIALS = itertools.count(1)
_QTBOT: list = []


@pytest.fixture(autouse=True)
def _own_qt_controls(qtbot):
    """Give every Qt control a test builds to ``qtbot``, which closes it."""
    _QTBOT.append(qtbot)
    yield
    _QTBOT.clear()


@pytest.fixture
def controller(qtbot):
    controller = CellierController(gui="offscreen")
    yield controller
    controller.close()


def _add_points(controller, names="zyx", name="points"):
    system = DataCoordinateSystem(
        name="data",
        datastore_id=uuid4(),
        axes=tuple(Axis(name=n, axis_type="space") for n in names),
    )
    highs = {"t": 9.0, "c": 1.0, "z": 10.0, "y": 20.0, "x": 40.0}
    positions = np.array(
        [[0.0] * len(names), [highs[n] for n in names]], dtype=np.float32
    )
    store = PointsMemoryStore(positions=positions, data_coordinate_systems=[system])
    scene = controller.add_scene(dim="3d", name=f"scene-{name}")
    visual = controller.add_points(data=store, scene_id=scene.id, name=name)
    return visual, store


def _make(toolkit, visual_ids, visual, store):
    seed = clipping_planes_seed(visual, store)
    if toolkit == "qt":
        from cellier.gui.qt.visuals import QtClippingPlanesControls

        control = QtClippingPlanesControls(visual_ids, **seed)
        _QTBOT[-1].addWidget(control.widget)
        return control
    from cellier.gui.anywidget.visuals import AnywidgetClippingPlanesControls

    return AnywidgetClippingPlanesControls(visual_ids, **seed)


def _act(widget, action, index=None, value=None) -> None:
    """Do one thing as a user would."""
    if hasattr(widget, "comm"):  # anywidget: what the front end sends
        widget.edit = {
            "action": action,
            "index": index,
            "value": value,
            "serial": next(_SERIALS),
        }
        return
    if action == "add":
        widget._add.click()
        return
    row = widget.row(index)
    if action == "remove":
        row.remove.click()
    elif action == "flip":
        row.flip.click()
    elif action == "enabled":
        row.enabled.setChecked(value)
    elif action == "facing":
        row.facing[tuple(value)].click()
    elif action == "position":
        row.position.setValue(value)
    elif action == "component":
        row.components[value[0]].setValue(value[1])


def _set_normal(widget, index, normal) -> None:
    """Enter *normal* one entry at a time, as a user would."""
    for axis, entry in enumerate(normal):
        _act(widget, "component", index, [axis, entry])


def _n_rows(widget) -> int:
    return len(widget.rows) if hasattr(widget, "comm") else len(widget._rows)


# -- the toolkit-free half ------------------------------------------------------


def test_rows_and_planes_convert_both_ways():
    system = uuid4()
    rows = [
        {"enabled": True, "normal": [0.0, 0.0, 2.0], "position": 12.0},
        {"enabled": False, "normal": [1.0, 1.0, 0.0], "position": -3.0},
    ]
    planes = planes_from_rows(rows, system)
    assert planes[0].plane.offset == 24.0  # position times the normal's length
    assert planes[1].enabled is False
    assert rows_from_planes(planes) == pytest.approx(rows)


def test_the_position_range_is_the_box_projected_on_the_normal():
    bounds = [[0, 10], [0, 20], [0, 40]]
    assert position_range([0, 0, 1], bounds) == (0.0, 40.0)
    assert position_range([0, 0, -2], bounds) == (-40.0, 0.0)
    low, high = position_range([0, 1, 1], bounds)
    assert (low, high) == pytest.approx((0.0, 60 / np.sqrt(2)))


def test_the_facing_of_a_normal():
    assert facing_of([0, -3, 0]) == [1, -1]
    assert facing_of([0, 0, 2]) == [2, 1]
    assert facing_of([1, 1, 0]) is None
    assert facing_of([0, 0, 0]) is None


def test_the_seed_is_read_off_the_store(controller):
    visual, store = _add_points(controller)
    seed = clipping_planes_seed(visual, store)
    assert seed["axis_names"] == ["z", "y", "x"]
    assert seed["bounds"] == [[0.0, 10.0], [0.0, 20.0], [0.0, 40.0]]
    assert seed["coordinate_system"] == str(store.data_coordinate_system.id)
    assert seed["data_store_id"] == str(store.id)
    assert seed["planes"] == []


# -- the control ----------------------------------------------------------------


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_it_adds_moves_toggles_and_removes_planes(controller, toolkit):
    visual, store = _add_points(controller)
    widget = _make(toolkit, [visual.id], visual, store)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    system = store.data_coordinate_system

    _act(widget, "add")
    # Across the last axis, through the middle of the data.
    assert visual.clipping_planes == (
        ClippingPlane.from_point_normal(system, (0, 0, 20), (0, 0, 1)),
    )
    _act(widget, "position", 0, 5.0)
    assert visual.clipping_planes[0].plane.offset == 5.0
    _act(widget, "flip", 0)
    np.testing.assert_array_equal(visual.clipping_planes[0].plane.normal, [0, 0, -1])
    assert visual.clipping_planes[0].plane.offset == -5.0  # the same plane
    _act(widget, "enabled", 0, False)
    assert visual.clipping_planes[0].enabled is False
    assert _n_rows(widget) == 1  # a disabled plane stays in the list

    _act(widget, "add")
    assert len(visual.clipping_planes) == 2
    _act(widget, "facing", 1, [1, 1])
    np.testing.assert_array_equal(visual.clipping_planes[1].plane.normal, [0, 1, 0])
    _act(widget, "facing", 1, [0, -1])
    np.testing.assert_array_equal(visual.clipping_planes[1].plane.normal, [-1, 0, 0])
    # An oblique normal, an entry at a time; z last, so it is never all zero.
    for axis, entry in ((1, 1.0), (2, 1.0), (0, 0.0)):
        _act(widget, "component", 1, [axis, entry])
    np.testing.assert_array_equal(visual.clipping_planes[1].plane.normal, [0, 1, 1])

    _act(widget, "remove", 0)
    assert len(visual.clipping_planes) == 1
    assert _n_rows(widget) == 1
    assert widget.error == ""
    widget.close()


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_turning_the_normal_keeps_the_plane_through_the_middle(controller, toolkit):
    visual, store = _add_points(controller)
    widget = _make(toolkit, [visual.id], visual, store)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    _act(widget, "add")
    _act(widget, "facing", 0, [0, 1])
    plane = visual.clipping_planes[0].plane
    # The data's centre is (5, 10, 20); the plane still passes through it.
    assert plane.signed_distance([5.0, 10.0, 20.0]) == pytest.approx(0.0)
    widget.close()


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_a_change_from_elsewhere_is_shown_and_sends_nothing(controller, toolkit):
    visual, store = _add_points(controller)
    widget = _make(toolkit, [visual.id], visual, store)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    sent: list = []
    widget.changed.connect(sent.append)
    system = store.data_coordinate_system
    visual.clipping_planes = (
        ClippingPlane.from_point_normal(system, (0, 7, 0), (0, 1, 0)),
        ClippingPlane.from_point_normal(system, (0, 0, 9), (0, 0, 1), enabled=False),
    )
    assert _n_rows(widget) == 2
    assert widget.editor.rows[0]["position"] == 7.0
    assert widget.editor.rows[1]["enabled"] is False
    assert sent == []
    widget.close()


def test_the_qt_row_labels_the_normal_with_the_stores_axis_names(controller):
    visual, store = _add_points(controller, names="tzyx")
    widget = _make("qt", [visual.id], visual, store)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    _act(widget, "add")
    row = widget.row(0)
    assert [button.text() for button in row.facing.values()] == [
        "+t",
        "-t",
        "+z",
        "-z",
        "+y",
        "-y",
        "+x",
        "-x",
    ]
    assert len(row.components) == 4
    assert [c.value() for c in row.components] == [0.0, 0.0, 0.0, 1.0]


def test_the_qt_facing_buttons_set_the_axis_and_the_kept_side(controller):
    visual, store = _add_points(controller)
    widget = _make("qt", [visual.id], visual, store)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    _act(widget, "add")
    row = widget.row(0)

    def checked():
        return [key for key, button in row.facing.items() if button.isChecked()]

    assert checked() == [(2, 1)]
    row.facing[(1, -1)].click()
    np.testing.assert_array_equal(visual.clipping_planes[0].plane.normal, [0, -1, 0])
    assert checked() == [(1, -1)]
    # The plane turned in place: it still passes through the data's centre.
    centre = [5.0, 10.0, 20.0]
    assert visual.clipping_planes[0].plane.signed_distance(centre) == pytest.approx(0)
    # A second click on the shown button changes nothing and stays checked.
    row.facing[(1, -1)].click()
    assert checked() == [(1, -1)]
    # An oblique normal lies along no axis.
    row.components[2].setValue(0.5)
    np.testing.assert_array_equal(visual.clipping_planes[0].plane.normal, [0, -1, 0.5])
    assert checked() == []
    assert facing_of([0, -1, 0.5]) is None


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_a_zeroed_normal_is_refused_with_a_reason(controller, toolkit):
    visual, store = _add_points(controller)
    widget = _make(toolkit, [visual.id], visual, store)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    _act(widget, "add")
    before = visual.clipping_planes
    _act(widget, "component", 0, [2, 0.0])
    assert visual.clipping_planes == before
    assert "zero" in widget.error
    if toolkit == "qt":  # the entry is put back
        assert widget.row(0).components[2].value() == 1.0
    else:
        assert widget.rows[0]["normal"] == [0.0, 0.0, 1.0]
    _act(widget, "component", 0, [1, 2.0])
    assert widget.error == ""
    widget.close()


def test_the_anywidget_ignores_a_malformed_edit(controller):
    visual, store = _add_points(controller)
    widget = _make("anywidget", [visual.id], visual, store)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    _act(widget, "add")
    before = visual.clipping_planes
    for value in (None, [7, 1], ["x", 1], [1]):
        _act(widget, "facing", 0, value)
        _act(widget, "component", 0, value)
    assert visual.clipping_planes == before
    widget.close()


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_the_position_range_follows_the_stores_extent(controller, toolkit):
    from cellier.convenience import PointsControlsConfig
    from cellier.convenience.layout._shared import appearance_specs

    visual, store = _add_points(controller)
    config = PointsControlsConfig(appearance=True, clipping_controls=True)
    spec = next(
        spec
        for spec in appearance_specs(visual, config, store).specs
        if spec.kind == "clipping_planes"
    )
    if toolkit == "qt":
        from cellier.convenience.gui._appearance_widgets_qt import QT_BUILDERS

        widget = QT_BUILDERS["clipping_planes"](spec, [visual.id], controller)
        _QTBOT[-1].addWidget(widget.widget)
    else:
        from cellier.convenience.gui._appearance_widgets import ANYWIDGET_BUILDERS

        widget = ANYWIDGET_BUILDERS["clipping_planes"](spec, [visual.id], controller)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    _act(widget, "add")
    described = widget.editor.describe()[0]
    assert (described["low"], described["high"]) == (0.0, 40.0)

    sent: list = []
    widget.changed.connect(sent.append)
    store.positions = np.array([[0, 0, 0], [10, 20, 80]], dtype=np.float32)
    described = widget.editor.describe()[0]
    assert (described["low"], described["high"]) == (0.0, 80.0)
    assert sent == []  # the plane did not move
    widget.close()


def test_a_moved_plane_keeps_its_row_widgets(controller):
    """A slider must not be destroyed while it is dragged."""
    visual, store = _add_points(controller)
    widget = _make("qt", [visual.id], visual, store)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    _act(widget, "add")
    slider = widget.row(0).slider
    for step in (100, 400, 900):
        slider.setValue(step)
        assert widget.row(0).slider is slider
        assert visual.clipping_planes[0].plane.offset == pytest.approx(40 * step / 1000)


def test_the_anywidget_sends_one_update_for_an_edit_delivered_twice(controller):
    """marimo delivers a front-end edit twice."""
    visual, store = _add_points(controller)
    widget = _make("anywidget", [visual.id], visual, store)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    _act(widget, "add")
    sent: list = []
    widget.changed.connect(sent.append)
    _act(widget, "position", 0, 5.0)
    _act(widget, "position", 0, 5.0)
    assert len(sent) == 1
    # What the front end draws a row from.
    assert widget.rows[0]["facing"] == [2, 1]
    assert (widget.rows[0]["low"], widget.rows[0]["high"]) == (0.0, 40.0)
    assert widget.axis_names == ["z", "y", "x"]
    widget.close()


def test_the_anywidget_puts_back_rows_a_host_overwrites(controller):
    """marimo sends the whole state with an edit, stale ``rows`` included.

    Found in a browser: a flipped plane kept its old normal and range on
    screen because the front end's copy of ``rows`` came back with the edit.
    """
    visual, store = _add_points(controller)
    widget = _make("anywidget", [visual.id], visual, store)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    _act(widget, "add")
    stale = [dict(row) for row in widget.rows]
    _act(widget, "flip", 0)
    flipped = [dict(row) for row in widget.rows]
    assert flipped[0]["normal"] == [0.0, 0.0, -1.0]

    widget.rows = stale  # what the host writes after the edit
    assert widget.rows == flipped
    widget.error = "left over"
    assert widget.error == ""
    np.testing.assert_array_equal(visual.clipping_planes[0].plane.normal, [0, 0, -1])
    widget.close()


def test_a_group_control_edits_every_visual(controller):
    first, store = _add_points(controller, name="a")
    scene = controller.add_scene(dim="2d", name="second view")
    second = controller.add_points(data=store, scene_id=scene.id, name="b")
    widget = _make("qt", [first.id, second.id], first, store)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    _act(widget, "add")
    assert first.clipping_planes == second.clipping_planes
    assert len(first.clipping_planes) == 1


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_a_control_for_five_axes(controller, toolkit):
    visual, store = _add_points(controller, names="tczyx")
    widget = _make(toolkit, [visual.id], visual, store)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    _act(widget, "add")
    np.testing.assert_array_equal(
        visual.clipping_planes[0].plane.normal, [0, 0, 0, 0, 1]
    )
    # An oblique normal over z, y and x: t and c are not constrained.
    _set_normal(widget, 0, [0, 0, 1, 1, 1])
    assert widget.error == ""
    described = widget.editor.describe()[0]
    assert described["normal"] == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert described["facing"] is None
    assert (described["low"], described["high"]) == pytest.approx(
        (0.0, (10 + 20 + 40) / np.sqrt(3))
    )
    widget.close()


def test_both_toolkits_carry_the_shared_title():
    from cellier.gui.anywidget.visuals import AnywidgetClippingPlanesControls
    from cellier.gui.qt.visuals import QtClippingPlanesControls

    assert QtClippingPlanesControls.DEFAULT_TITLE == CLIPPING_PLANES_TITLE
    assert AnywidgetClippingPlanesControls.DEFAULT_TITLE == "Clipping planes"


# -- the panel ------------------------------------------------------------------


def test_the_panel_offers_it_only_when_asked(controller):
    from cellier.convenience import PointsControlsConfig
    from cellier.convenience.layout._shared import appearance_specs

    visual, store = _add_points(controller)

    def kinds(config):
        return [spec.kind for spec in appearance_specs(visual, config, store).specs]

    assert "clipping_planes" not in kinds(PointsControlsConfig(appearance=True))
    asked = PointsControlsConfig(appearance=True, clipping_controls=True)
    assert "clipping_planes" in kinds(asked)
    spec = next(
        spec
        for spec in appearance_specs(visual, asked, store).specs
        if spec.kind == "clipping_planes"
    )
    assert spec.title == "Clipping planes"
    assert spec.values == clipping_planes_seed(visual, store)


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_the_panel_builders_make_the_control(controller, toolkit):
    from cellier.convenience import PointsControlsConfig
    from cellier.convenience.layout._shared import appearance_specs

    visual, store = _add_points(controller)
    visual.clipping_planes = (
        ClippingPlane.from_point_normal(
            store.data_coordinate_system, (0, 0, 9), (0, 0, 1)
        ),
    )
    config = PointsControlsConfig(appearance=True, clipping_controls=True)
    spec = next(
        spec
        for spec in appearance_specs(visual, config, store).specs
        if spec.kind == "clipping_planes"
    )
    if toolkit == "qt":
        from cellier.convenience.gui._appearance_widgets_qt import QT_BUILDERS

        widget = QT_BUILDERS["clipping_planes"](spec, [visual.id], controller)
        _QTBOT[-1].addWidget(widget.widget)
    else:
        from cellier.convenience.gui._appearance_widgets import ANYWIDGET_BUILDERS

        widget = ANYWIDGET_BUILDERS["clipping_planes"](spec, [visual.id], controller)
    assert _n_rows(widget) == 1
    assert widget.editor.rows[0]["position"] == 9.0
