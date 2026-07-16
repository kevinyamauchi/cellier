"""Tests for canvas teardown: ``CanvasView.close`` and its callers.

A render canvas is a parentless widget owned by the GUI backend, so it is not
reclaimed by dropping Python references -- through its draw callback and event
filter it pins the ``CanvasView``, the ``WgpuRenderer``, and everything they
reach.  Only ``close()`` breaks that chain.  These tests pin the release down
with weak references, because a regression here is invisible (nothing fails --
the suite just accumulates live canvases and slows down until CI dies).
"""

from __future__ import annotations

import gc
import weakref

import pytest

from cellier.controller import CellierController
from cellier.scene.dims import CoordinateSystem


def _controller_with_canvas(qtbot) -> CellierController:
    """A controller with one 2-D scene and one attached canvas."""
    controller = CellierController()
    scene = controller.add_scene(
        dim="2d",
        coordinate_system=CoordinateSystem(name="world", axis_labels=("y", "x")),
        name="main",
        render_modes={"2d"},
    )
    controller.add_canvas(scene.id)
    return controller


def _only_canvas_view(controller: CellierController):
    return next(iter(controller._render_manager._canvases.values()))


def test_close_releases_canvas_view_and_widget(qtbot):
    controller = _controller_with_canvas(qtbot)
    view = _only_canvas_view(controller)
    view_ref = weakref.ref(view)
    widget_ref = weakref.ref(view.widget)

    controller.close()
    del controller, view
    gc.collect()

    assert view_ref() is None
    assert widget_ref() is None


def test_canvas_view_is_not_released_without_close(qtbot):
    """The counterpart to the test above: dropping references is not enough.

    Documents *why* close() must exist -- if this ever starts failing, the GUI
    backend no longer owns the canvas and the explicit teardown could go.
    """
    controller = _controller_with_canvas(qtbot)
    view_ref = weakref.ref(_only_canvas_view(controller))

    del controller
    gc.collect()

    assert view_ref() is not None
    view = view_ref()
    view.close()  # don't leak out of this test


def test_close_is_idempotent(qtbot):
    controller = _controller_with_canvas(qtbot)
    view = _only_canvas_view(controller)

    view.close()
    view.close()

    assert view._closed


def test_close_survives_widget_already_destroyed_by_qt(qtbot):
    """Qt may destroy the widget first (e.g. the user closed the window)."""
    controller = _controller_with_canvas(qtbot)
    view = _only_canvas_view(controller)

    view.widget.close()
    view.widget.deleteLater()
    qtbot.wait(10)

    view.close()  # must not raise

    assert view._closed


def test_draw_frame_after_close_is_a_no_op(qtbot):
    """A draw queued with the backend can still arrive after close()."""
    controller = _controller_with_canvas(qtbot)
    view = _only_canvas_view(controller)
    view.close()

    view._draw_frame()  # must not touch the released surface


def test_controller_close_closes_every_canvas(qtbot):
    controller = CellierController()
    scene = controller.add_scene(
        dim="2d",
        coordinate_system=CoordinateSystem(name="world", axis_labels=("y", "x")),
        name="main",
        render_modes={"2d"},
    )
    controller.add_canvas(scene.id)
    controller.add_canvas(scene.id)
    views = list(controller._render_manager._canvases.values())
    assert len(views) == 2

    controller.close()

    assert all(view._closed for view in views)
    assert controller._render_manager._canvases == {}


def test_remove_canvas_closes_the_canvas(qtbot):
    controller = _controller_with_canvas(qtbot)
    canvas_id = next(iter(controller._render_manager._canvases))
    view = controller._render_manager._canvases[canvas_id]

    controller._render_manager.remove_canvas(canvas_id)

    assert view._closed
    assert canvas_id not in controller._render_manager._canvases


def test_remove_scene_closes_its_canvases(qtbot):
    controller = _controller_with_canvas(qtbot)
    scene_id = next(iter(controller._scene_to_canvases))
    view = _only_canvas_view(controller)

    controller._render_manager.remove_scene(scene_id)

    assert view._closed
    assert controller._render_manager._canvases == {}


@pytest.mark.parametrize("gui", ["qt", "anywidget"])
def test_close_releases_canvas_for_both_guis(qtbot, gui):
    """The anywidget backend closes re-entrantly; close() must still terminate.

    rendercanvas's anywidget ``_rc_close`` dispatches a synthetic "close"
    message whose handler calls ``close()`` again, so an unguarded close
    recurses until the stack blows.
    """
    pytest.importorskip("anywidget")

    controller = CellierController(gui=gui)
    scene = controller.add_scene(
        dim="2d",
        coordinate_system=CoordinateSystem(name="world", axis_labels=("y", "x")),
        name="main",
        render_modes={"2d"},
    )
    controller.add_canvas(scene.id)
    view_ref = weakref.ref(_only_canvas_view(controller))

    controller.close()
    del controller
    gc.collect()

    assert view_ref() is None
