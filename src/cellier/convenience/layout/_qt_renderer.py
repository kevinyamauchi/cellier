"""Qt renderer -- the view layer for Layout specs on the Qt backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellier.convenience.layout._spec import Layout

# render_mode, iso_threshold, and attenuation are handled by one shared widget.
_RENDER_FIELDS = frozenset({"render_mode", "iso_threshold", "attenuation"})


def render_qt(layout: Layout, viewer: object) -> object:
    """Render a Layout spec to a ``QMainWindow``.

    Builds the center widget from *layout.center* using Qt layout primitives,
    sets it as the central widget, and wraps each non-None dock spec in a
    ``QDockWidget``.

    Parameters
    ----------
    layout : Layout
        The layout spec to render.
    viewer :
        The viewer, reserved for future scene-level control building.

    Returns
    -------
    QMainWindow
    """
    from PySide6 import QtWidgets
    from PySide6.QtCore import Qt

    window = QtWidgets.QMainWindow()
    window.setCentralWidget(_render_center_qt(layout.center))

    dock_map = {
        "left": (layout.left_dock, Qt.DockWidgetArea.LeftDockWidgetArea),
        "right": (layout.right_dock, Qt.DockWidgetArea.RightDockWidgetArea),
        "top": (layout.top_dock, Qt.DockWidgetArea.TopDockWidgetArea),
        "bottom": (layout.bottom_dock, Qt.DockWidgetArea.BottomDockWidgetArea),
    }
    for name, (spec, area) in dock_map.items():
        widget = _render_dock_qt(spec, viewer)
        if widget is not None:
            widget = _wrap_dock_widget(widget, name)
            dock = QtWidgets.QDockWidget(name.capitalize(), window)
            dock.setWidget(widget)
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            )
            window.addDockWidget(area, dock)

    from PySide6.QtWidgets import QApplication

    screen = QApplication.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        w = min(int(available.width() * 2 / 3), 1600)
        h = min(int(available.height() * 2 / 3), 1000)
        window.resize(w, h)

    return window


def _wrap_dock_widget(widget: object, position: str) -> object:
    """Center *widget* in a stretch container sized for *position*.

    Top/bottom docks: horizontal container (stretch | widget | stretch).
    Left/right docks: vertical container (stretch / widget / stretch).
    """
    from PySide6 import QtWidgets

    container = QtWidgets.QWidget()
    if position in ("top", "bottom"):
        box = QtWidgets.QHBoxLayout(container)
    else:
        box = QtWidgets.QVBoxLayout(container)
    box.setContentsMargins(4, 4, 4, 4)
    box.addStretch()
    box.addWidget(widget)
    box.addStretch()
    return container


def _render_center_qt(node: object) -> object:
    """Recursively render a center spec node to a Qt widget."""
    from PySide6 import QtWidgets

    from cellier.convenience.layout._spec import Grid, HStack, VStack

    if isinstance(node, HStack):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for item in node.items:
            layout.addWidget(_render_center_qt(item))
        return container

    if isinstance(node, VStack):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for item in node.items:
            layout.addWidget(_render_center_qt(item))
        return container

    if isinstance(node, Grid):
        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        for row_idx, row in enumerate(node.cells):
            for col_idx, cell in enumerate(row):
                if cell is not None:
                    grid.addWidget(_render_center_qt(cell), row_idx, col_idx)
        return container

    # Leaf: QtCanvasWidget or OrthoCanvasWidgets -- both expose .widget.
    if hasattr(node, "widget"):
        return node.widget
    raise TypeError(
        f"Cannot render {type(node).__name__!r} as a Qt center widget. "
        "Expected a QtCanvasWidget, HStack, VStack, or Grid."
    )


def _render_scene_controls_qt(viewer: object) -> object | None:
    """Build a Qt scene controls widget (2D/3D dim toggle)."""
    from cellier.gui.qt._toggle import make_dim_toggle_qt

    return make_dim_toggle_qt(viewer)


def _colormap_to_str(cm) -> str:
    if isinstance(cm, str):
        return cm
    name = getattr(cm, "name", None)
    if name is not None and isinstance(name, str):
        return name
    return str(cm)


def _render_appearance_controls_qt(viewer: object) -> object | None:
    """Build and wire Qt appearance sub-widgets for the first configured visual.

    Mirrors _render_appearance_controls in _anywidget_renderer.py: reads
    viewer._controls_configs, finds the visual, instantiates each requested
    sub-widget from cellier.gui.qt.visuals, wires each to the controller,
    and returns a QWidget container.
    """
    from PySide6 import QtWidgets
    from PySide6.QtWidgets import QSizePolicy

    from cellier.convenience.gui._controls_config import InMemoryImageControlsConfig
    from cellier.gui.qt.visuals import (
        QtClimRangeSlider,
        QtColormapComboBox,
        QtLodBiasSlider,
        QtVolumeRenderControls,
    )

    controls_configs: dict = getattr(viewer, "_controls_configs", {})
    scene = getattr(viewer, "scene", None)
    if scene is None or not controls_configs:
        return None

    controls_config = None
    visual = None
    for v in scene.visuals:
        if v.id in controls_configs:
            visual = v
            controls_config = controls_configs[v.id]
            break

    if controls_config is None:
        return None

    field_list = (
        controls_config.appearance
        if isinstance(controls_config.appearance, list) and controls_config.appearance
        else None
    )
    if not field_list or not hasattr(visual, "appearance"):
        return None

    fields = set(field_list)
    app = visual.appearance

    raw_clim = tuple(getattr(app, "clim", (0.0, 1.0)))
    if (
        isinstance(controls_config, InMemoryImageControlsConfig)
        and controls_config.clim_range is not None
    ):
        clim_range: tuple[float, float] = controls_config.clim_range
    else:
        clim_range = (min(0.0, float(raw_clim[0])), max(1.0, float(raw_clim[1])))

    colormap_names = (
        controls_config.colormap_names
        if isinstance(controls_config, InMemoryImageControlsConfig)
        else None
    )

    container = QtWidgets.QWidget()
    container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
    container.setMinimumWidth(240)
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(6)

    def _group(title: str, widget: object) -> None:
        grp = QtWidgets.QGroupBox(title)
        box = QtWidgets.QVBoxLayout(grp)
        box.setContentsMargins(12, 4, 12, 4)
        box.addWidget(widget)
        layout.addWidget(grp)

    if "color_map" in fields and hasattr(app, "color_map"):
        combo = QtColormapComboBox(
            visual.id,
            initial_colormap=_colormap_to_str(getattr(app, "color_map", "grays")),
        )
        if colormap_names is not None:
            combo.add_colormaps(colormap_names)
        viewer.controller.connect_widget(
            combo, subscription_specs=combo.subscription_specs()
        )
        _group("Colormap", combo.widget)

    if "clim" in fields and hasattr(app, "clim"):
        clim_w = QtClimRangeSlider(
            visual.id,
            clim_range=clim_range,
            initial_clim=raw_clim,
        )
        viewer.controller.connect_widget(
            clim_w, subscription_specs=clim_w.subscription_specs()
        )
        _group("Contrast limits", clim_w.widget)

    if fields & _RENDER_FIELDS and any(hasattr(app, f) for f in _RENDER_FIELDS):
        render_w = QtVolumeRenderControls(
            visual.id,
            dtype_max=float(clim_range[1]),
            initial_render_mode=getattr(app, "render_mode", "mip"),
            initial_threshold=getattr(app, "iso_threshold", 0.2),
            initial_attenuation=getattr(app, "attenuation", 1.0),
        )
        viewer.controller.connect_widget(
            render_w, subscription_specs=render_w.subscription_specs()
        )
        _group("Render mode", render_w.widget)

    if "lod_bias" in fields and hasattr(app, "lod_bias"):
        lod_w = QtLodBiasSlider(
            visual.id,
            initial_lod_bias=float(getattr(app, "lod_bias", 1.0)),
        )
        viewer.controller.connect_widget(
            lod_w, subscription_specs=lod_w.subscription_specs()
        )
        _group("LOD bias", lod_w.widget)

    layout.addStretch()
    return container


def _render_dock_qt(spec: object, viewer: object) -> object | None:
    """Render one dock spec to a Qt widget, or return None."""
    if spec is None:
        return None

    from PySide6 import QtWidgets

    from cellier.convenience.layout._spec import (
        AppearanceControls,
        HStack,
        SceneControls,
        VStack,
    )

    if isinstance(spec, AppearanceControls):
        return _render_appearance_controls_qt(viewer)
    if isinstance(spec, SceneControls):
        return _render_scene_controls_qt(viewer)
    if isinstance(spec, (HStack, VStack)):
        container = QtWidgets.QWidget()
        box = (
            QtWidgets.QHBoxLayout(container)
            if isinstance(spec, HStack)
            else QtWidgets.QVBoxLayout(container)
        )
        box.setContentsMargins(4, 4, 4, 4)
        for item in spec.items:
            widget = _render_dock_qt(item, viewer)
            if widget is not None:
                box.addWidget(widget)
        return container
    return None
