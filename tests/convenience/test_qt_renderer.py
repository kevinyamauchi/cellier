"""Tests for ``cellier.convenience.layout._qt_renderer``.

Closes the Qt-renderer parity hole (the anywidget renderer is covered by
``tests/v2/test_anywidget.py``). The appearance-controls builder is exercised
against a real ``Viewer`` + controller; the center/window builders use fake
leaf widgets (any object exposing ``.widget``) so no render surface is built.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

from cellier.convenience import Viewer  # noqa: E402
from cellier.convenience.layout._qt_renderer import (  # noqa: E402
    _render_appearance_controls_qt,
    _render_center_qt,
    _render_dock_qt,
    _wrap_dock_widget,
    render_qt,
)
from cellier.convenience.layout._spec import (  # noqa: E402
    AppearanceControls,
    Grid,
    HStack,
    Layout,
    VStack,
)
from cellier.visuals._image import MultiscaleImageAppearance  # noqa: E402
from cellier.visuals._image_memory import InMemoryImageAppearance  # noqa: E402


def _leaf():
    from PySide6 import QtWidgets

    return SimpleNamespace(widget=QtWidgets.QLabel())


def _group_titles(container):
    from PySide6 import QtWidgets

    return {g.title() for g in container.findChildren(QtWidgets.QGroupBox)}


# ---------------------------------------------------------------------------
# _render_appearance_controls_qt
# ---------------------------------------------------------------------------


def test_appearance_controls_builds_colormap_and_clim_groups(qtbot, image_store):
    viewer = Viewer(("z", "y", "x"), gui="qt")
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(color_map="grays", clim=(0.0, 1.0)),
        controls={"appearance": ["color_map", "clim"]},
    )

    container = _render_appearance_controls_qt(viewer)

    assert container is not None
    assert {"Colormap", "Contrast limits"} <= _group_titles(container)


def test_appearance_controls_explicit_clim_range(qtbot, image_store):
    viewer = Viewer(("z", "y", "x"), gui="qt")
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(color_map="grays", clim=(0.0, 1.0)),
        controls={"appearance": ["clim"], "clim_range": (0.0, 5.0)},
    )

    container = _render_appearance_controls_qt(viewer)

    assert container is not None
    assert "Contrast limits" in _group_titles(container)


def test_appearance_controls_multiscale_render_and_lod(qtbot, multiscale_image_store):
    viewer = Viewer(("z", "y", "x"), gui="qt")
    viewer.add_image_multiscale(
        multiscale_image_store,
        appearance=MultiscaleImageAppearance(color_map="viridis", render_mode="mip"),
        controls={"appearance": ["render_mode", "lod_bias"]},
    )

    container = _render_appearance_controls_qt(viewer)

    assert container is not None
    assert {"Render mode", "LOD bias"} <= _group_titles(container)


def test_appearance_controls_none_without_configs(qtbot, image_store):
    viewer = Viewer(("z", "y", "x"), gui="qt")
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    assert _render_appearance_controls_qt(viewer) is None


# ---------------------------------------------------------------------------
# _render_center_qt
# ---------------------------------------------------------------------------


def test_render_center_leaf_returns_inner_widget(qtbot):
    leaf = _leaf()
    assert _render_center_qt(leaf) is leaf.widget


def test_render_center_hstack(qtbot):
    from PySide6 import QtWidgets

    container = _render_center_qt(HStack(items=[_leaf()]))
    assert isinstance(container.layout(), QtWidgets.QHBoxLayout)


def test_render_center_vstack(qtbot):
    from PySide6 import QtWidgets

    container = _render_center_qt(VStack(items=[_leaf()]))
    assert isinstance(container.layout(), QtWidgets.QVBoxLayout)


def test_render_center_grid(qtbot):
    from PySide6 import QtWidgets

    container = _render_center_qt(Grid(cells=[[_leaf(), None]]))
    assert isinstance(container.layout(), QtWidgets.QGridLayout)


def test_render_center_unrenderable_raises(qtbot):
    with pytest.raises(TypeError, match="Cannot render"):
        _render_center_qt(object())


# ---------------------------------------------------------------------------
# _wrap_dock_widget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "position, layout_cls",
    [("top", "QHBoxLayout"), ("bottom", "QHBoxLayout"), ("left", "QVBoxLayout")],
)
def test_wrap_dock_widget_orientation(qtbot, position, layout_cls):
    from PySide6 import QtWidgets

    container = _wrap_dock_widget(QtWidgets.QLabel(), position)
    assert type(container.layout()).__name__ == layout_cls


# ---------------------------------------------------------------------------
# _render_dock_qt
# ---------------------------------------------------------------------------


def test_render_dock_none_returns_none(qtbot, image_store):
    viewer = Viewer(("z", "y", "x"), gui="qt")
    assert _render_dock_qt(None, viewer) is None


def test_render_dock_stack_of_appearance(qtbot, image_store):
    viewer = Viewer(("z", "y", "x"), gui="qt")
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(color_map="grays", clim=(0.0, 1.0)),
        controls={"appearance": ["color_map"]},
    )

    rendered = _render_dock_qt(VStack(items=[AppearanceControls()]), viewer)

    assert rendered is not None


# ---------------------------------------------------------------------------
# render_qt
# ---------------------------------------------------------------------------


def test_render_qt_builds_window_with_dock(qtbot, image_store):
    from PySide6 import QtWidgets

    viewer = Viewer(("z", "y", "x"), gui="qt")
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(color_map="grays", clim=(0.0, 1.0)),
        controls={"appearance": ["color_map", "clim"]},
    )
    leaf = _leaf()
    layout = Layout(center=leaf, right_dock=AppearanceControls())

    window = render_qt(layout, viewer)

    assert isinstance(window, QtWidgets.QMainWindow)
    assert window.centralWidget() is leaf.widget
    assert len(window.findChildren(QtWidgets.QDockWidget)) == 1
