"""Tests for the Qt ``QtDimsControl`` widget (slider panel + 2D/3D toggle)."""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

from cellier._state import AxisAlignedSelectionState, DimsState  # noqa: E402
from cellier.controller import CellierController  # noqa: E402
from cellier.events import DimsChangedEvent  # noqa: E402
from cellier.gui.qt._scene import QtDimsControl  # noqa: E402
from cellier.scene.dims import CoordinateSystem  # noqa: E402


def _make_controller_with_scene(*, dim="2d"):
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(dim=dim, coordinate_system=cs, name="main")
    return controller, scene


def _make_control(scene, qtbot, *, with_toggle=True) -> QtDimsControl:
    selection = scene.dims.selection
    control = QtDimsControl(
        scene_id=scene.id,
        axis_ranges={0: (0, 9), 1: (0, 99), 2: (0, 99)},
        axis_labels={0: "z", 1: "y", 2: "x"},
        initial_slice_indices=dict(selection.slice_indices),
        initial_displayed_axes=selection.displayed_axes,
        axes_2d=(1, 2) if with_toggle else None,
        axes_3d=(0, 1, 2) if with_toggle else None,
    )
    qtbot.addWidget(control.widget)
    return control


def _dims_changed_event(source_id, scene_id, *, displayed, slices, stacked=()):
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


def test_slider_drag_updates_model(qtbot):
    controller, scene = _make_controller_with_scene(dim="2d")
    control = _make_control(scene, qtbot)
    controller.connect_widget(control, subscription_specs=control.subscription_specs())

    control._sliders[0].setValue(5)
    assert scene.dims.selection.slice_indices[0] == 5


def test_toggle_button_omitted_without_axes(qtbot):
    _controller, scene = _make_controller_with_scene(dim="2d")
    control = _make_control(scene, qtbot, with_toggle=False)
    assert control._toggle_button is None


def test_toggle_click_uses_live_slider_value_and_updates_model(qtbot):
    controller, scene = _make_controller_with_scene(dim="2d")
    control = _make_control(scene, qtbot)
    controller.connect_widget(control, subscription_specs=control.subscription_specs())

    assert control._toggle_button.text() == "Switch to 3D"

    # Move axis 0's slider before toggling to 3D -- it becomes displayed and
    # so has nothing to restore yet.
    control._sliders[0].setValue(5)
    assert scene.dims.selection.slice_indices[0] == 5

    control._on_toggle_click()  # 2D -> 3D
    assert len(scene.dims.selection.displayed_axes) == 3
    assert control._toggle_button.text() == "Switch to 2D"

    control._on_toggle_click()  # 3D -> 2D
    assert len(scene.dims.selection.displayed_axes) == 2
    assert control._toggle_button.text() == "Switch to 3D"
    # Regression guard: axis 0's slice value survives the round trip through
    # 3D instead of being reset to a hardcoded default.
    assert scene.dims.selection.slice_indices[0] == 5


def test_external_dims_change_resyncs_sliders_and_toggle(qtbot):
    controller, scene = _make_controller_with_scene(dim="2d")
    control = _make_control(scene, qtbot)
    controller.connect_widget(control, subscription_specs=control.subscription_specs())

    # Driven by something else entirely (not this widget's own button/slider).
    controller.set_displayed_axes(scene.id, (0, 1, 2))

    assert control._toggle_button.text() == "Switch to 2D"


def test_echoed_event_is_ignored(qtbot):
    _controller, scene = _make_controller_with_scene(dim="2d")
    control = _make_control(scene, qtbot)

    control._sliders[0].setValue(3)
    event = _dims_changed_event(
        source_id=control._id,
        scene_id=scene.id,
        displayed=(1, 2),
        slices={0: 999},
    )
    control._on_dims_changed(event)

    # An event stamped with our own id is our own echo; it must not reapply.
    assert control._sliders[0].value() == 3
