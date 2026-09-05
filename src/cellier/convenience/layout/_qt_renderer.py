"""Qt renderer -- the view layer for Layout specs on the Qt backend."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellier.convenience.layout._spec import Layout


class _CellierMainWindow:
    """Mixin giving a ``QMainWindow`` the teardown ``_RenderView`` provides.

    The anywidget path hands the caller a ``DisplayHandle`` whose ``close()``
    unsubscribes every control it built.  The Qt path hands back a window and
    nothing else, so its controls stayed subscribed to the bus for as long as
    the controller lived -- **and kept being delivered events** -- even after
    the window was closed.  Measured: building and dropping a Qt viewer left
    ~30 widgets and its controller alive per cycle, growing linearly.

    Closing the window now closes those controls, which is what makes them
    emit ``closed`` and the controller drop their subscriptions.  It
    deliberately does not close the *controller*: a window is a view, and the
    viewer may outlive it.  Releasing the canvases and the controller is
    ``CellierController.close()``, exactly as on the anywidget side.
    """

    def _cellier_init(self) -> None:
        self._cellier_closeables: list = []
        self._cellier_torn_down = False

    def _cellier_teardown(self) -> None:
        if self._cellier_torn_down:
            return
        self._cellier_torn_down = True
        for obj in self._cellier_closeables:
            close = getattr(obj, "close", None)
            if close is None:
                continue
            with suppress(Exception):
                close()
        self._cellier_closeables.clear()


def _make_window(QtWidgets):
    """Build the ``QMainWindow`` subclass, at call time so Qt stays optional."""

    class CellierMainWindow(_CellierMainWindow, QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self._cellier_init()

        def closeEvent(self, event) -> None:
            self._cellier_teardown()
            super().closeEvent(event)

    return CellierMainWindow


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

    window = _make_window(QtWidgets)()
    # The center is closed on teardown exactly as the docks are: a canvas
    # widget owns a dims control that is subscribed to the bus, and until this
    # list reached the center that control outlived every window that built it
    # (``tests/gui/test_backend_parity.py``).
    window.setCentralWidget(
        _render_center_qt(layout.center, window._cellier_closeables)
    )

    dock_map = {
        "left": (layout.left_dock, Qt.DockWidgetArea.LeftDockWidgetArea),
        "right": (layout.right_dock, Qt.DockWidgetArea.RightDockWidgetArea),
        "top": (layout.top_dock, Qt.DockWidgetArea.TopDockWidgetArea),
        "bottom": (layout.bottom_dock, Qt.DockWidgetArea.BottomDockWidgetArea),
    }
    for name, (spec, area) in dock_map.items():
        widget = _render_dock_qt(spec, viewer, window._cellier_closeables)
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


def _render_center_qt(node: object, closeables: list | None = None) -> object:
    """Recursively render a center spec node to a Qt widget.

    *closeables* collects every leaf that owns bus subscriptions, so closing
    the window releases them.  It mirrors the anywidget renderer's own list;
    it is optional only so the recursive calls and the tests that render a
    bare node need not pass one.
    """
    from PySide6 import QtWidgets

    from cellier.convenience.layout._spec import Grid, HStack, VStack

    if isinstance(node, HStack):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for item in node.items:
            layout.addWidget(_render_center_qt(item, closeables))
        return container

    if isinstance(node, VStack):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for item in node.items:
            layout.addWidget(_render_center_qt(item, closeables))
        return container

    if isinstance(node, Grid):
        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        for row_idx, row in enumerate(node.cells):
            for col_idx, cell in enumerate(row):
                if cell is not None:
                    grid.addWidget(
                        _render_center_qt(cell, closeables), row_idx, col_idx
                    )
        return container

    # Leaf: QtCanvasWidget or OrthoCanvasWidgets -- both expose .widget.
    if hasattr(node, "widget"):
        if closeables is not None and hasattr(node, "close"):
            closeables.append(node)
        return node.widget
    raise TypeError(
        f"Cannot render {type(node).__name__!r} as a Qt center widget. "
        "Expected a QtCanvasWidget, HStack, VStack, or Grid."
    )


# Each builder takes ``visual_ids`` -- one id on a ``Viewer``, the four panel
# siblings on an ``OrthoViewer``.  Every widget below accepts either shape via
# ``VisualIdGroup`` (design section 8.3 step 1), so the fan-out costs the
# builders nothing.


def _qt_color_map(spec, visual_ids, controller, parent):
    from cellier.gui.qt.visuals import QtColormapComboBox

    combo = QtColormapComboBox(
        visual_ids,
        initial_colormap=spec.values["initial_colormap"],
        title=spec.title,
    )
    names = spec.values["colormap_names"]
    if names is not None:
        combo.add_colormaps(names)
    return combo


def _qt_clim(spec, visual_ids, controller, parent):
    from cellier.gui.qt.visuals import QtClimRangeSlider

    return QtClimRangeSlider(
        visual_ids,
        clim_range=spec.values["clim_range"],
        initial_clim=spec.values["initial_clim"],
        title=spec.title,
    )


def _qt_render(spec, visual_ids, controller, parent):
    from cellier.gui.qt.visuals import QtVolumeRenderControls

    # ``dtype_max`` is a Qt-only construction keyword; the anywidget control
    # does not accept it.  Deriving it here from the neutral ``clim_range`` is
    # what keeps it out of the shared spec (design section 7.3).
    return QtVolumeRenderControls(
        visual_ids,
        dtype_max=float(spec.values["clim_range"][1]),
        initial_render_mode=spec.values["initial_render_mode"],
        initial_threshold=spec.values["initial_threshold"],
        initial_attenuation=spec.values["initial_attenuation"],
        title=spec.title,
    )


def _qt_lod_bias(spec, visual_ids, controller, parent):
    from cellier.gui.qt.visuals import QtLodBiasSlider

    return QtLodBiasSlider(
        visual_ids,
        initial_lod_bias=spec.values["initial_lod_bias"],
        title=spec.title,
    )


def _qt_aabb(spec, visual_ids, controller, parent):
    from cellier.gui.qt.visuals import QtAABBWidget

    return QtAABBWidget(
        visual_ids,
        initial_enabled=spec.values["initial_enabled"],
        initial_line_width=spec.values["initial_line_width"],
        initial_color=spec.values["initial_color"],
        title=spec.title,
    )


def _qt_field_control(spec, visual_ids, controller, parent):
    """Build any of the 23 single-field controls from the shared table.

    One builder rather than 22 dispatch entries: the layer-3 classes have a
    uniform constructor (``initial_value=``, plus ``choices=`` for a combo),
    so the only thing that varies is which class, and that is a table lookup
    (``cellier.gui._appearance_fields.APPEARANCE_FIELD_WIDGETS``).
    """
    from cellier.gui._appearance_fields import field_widget_class

    widget_class = field_widget_class(spec.kind, "qt")
    kwargs = {"initial_value": spec.values["initial_value"]}
    if "choices" in spec.values:
        kwargs["choices"] = spec.values["choices"]
    return widget_class(visual_ids, parent=parent, **kwargs)


def _qt_dataset_info(spec, visual_ids, controller, parent):
    """Build the read-only dataset-info block.

    Takes neither *visual_ids* nor *controller*: it displays what the spec
    handed it and drives nothing, which is why ``STATIC_CONTROL_KINDS`` keeps
    it off the bus.

    The spec carries either an ``info`` (a store's sectioned self-description)
    or flat ``rows`` (the hand-authored escape hatch); the widget has a
    constructor for each.
    """
    from cellier.gui.qt import QtDatasetInfo

    if "info" in spec.values:
        return QtDatasetInfo.from_info(
            spec.values["info"], title=spec.title, parent=parent
        )
    return QtDatasetInfo(spec.values["rows"], title=spec.title, parent=parent)


def _qt_visual_outline(spec, visual_ids, controller, parent):
    from cellier.gui.qt.render import QtVisualOutlineControls

    return QtVisualOutlineControls(
        visual_ids,
        spec.values,
        palette=spec.values.get("palette", ()),
        parent=parent,
    )


def _qt_labels_outline(spec, visual_ids, controller, parent):
    from cellier.gui.qt.render import QtLabelsOutlineControls

    return QtLabelsOutlineControls(
        visual_ids,
        spec.values,
        palette=spec.values.get("palette", ()),
        parent=parent,
    )


def _qt_visual_occlusion(spec, visual_ids, controller, parent):
    from cellier.gui.qt.render import QtVisualOcclusionControls

    return QtVisualOcclusionControls(visual_ids, spec.values, parent=parent)


def _qt_visual_picking(spec, visual_ids, controller, parent):
    from cellier.gui.qt.render import QtVisualPickingControls

    return QtVisualPickingControls(visual_ids, spec.values, parent=parent)


_QT_BUILDERS = {
    "color_map": _qt_color_map,
    "clim": _qt_clim,
    "render": _qt_render,
    "lod_bias": _qt_lod_bias,
    "aabb": _qt_aabb,
    "visual_outline": _qt_visual_outline,
    "labels_outline": _qt_labels_outline,
    "visual_occlusion": _qt_visual_occlusion,
    "visual_picking": _qt_visual_picking,
    "dataset_info": _qt_dataset_info,
}
"""``ControlSpec.kind`` -> Qt widget constructor."""


def _render_appearance_controls_qt(
    viewer: object, closeables: list | None = None
) -> object | None:
    """Build and wire the Qt appearance dock for the first configured visual.

    The decision of *which* controls, in what order, with what values is made
    by ``layout._shared.appearance_specs`` and shared with the anywidget path;
    this function is only the Qt view layer -- a dispatch table and a column.

    It draws no chrome of its own.  Each control names itself -- a label row
    for a single control, an owned ``QGroupBox`` for a block of rows that only
    means something together -- so this stacks them and stops
    (``plans/label_ownership_unification.md``).  Before that this function
    wrapped every control in a group box, which named the 18 of 23 field
    widgets that rendered bare and double-named the 5 that did not.
    """
    from PySide6 import QtWidgets
    from PySide6.QtWidgets import QSizePolicy

    from cellier.convenience.layout._shared import (
        APPEARANCE_DOCK_GAP_PX,
        STATIC_CONTROL_KINDS,
        _resolve_data_store,
        appearance_specs,
        select_appearance_target,
        warn_skipped_appearance_fields,
    )
    from cellier.gui._appearance_fields import APPEARANCE_FIELD_WIDGETS

    target = select_appearance_target(viewer)
    if target is None:
        return None

    specs, skipped = appearance_specs(
        target.visual,
        target.config,
        _resolve_data_store(getattr(viewer, "controller", None), target.visual),
        palette=viewer.controller.render_config.outline.palette,
    )
    warn_skipped_appearance_fields(skipped, target.visual, target.config)
    if not specs:
        # No requested field resolved to a control, so there is nothing to
        # dock.  Matches the anywidget path, and keeps ``appearance=True``
        # (still a dead value until stage 5) producing no panel.
        return None

    container = QtWidgets.QWidget()
    container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
    container.setMinimumWidth(260)
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(APPEARANCE_DOCK_GAP_PX)

    for spec in specs:
        builder = _QT_BUILDERS.get(spec.kind)
        if builder is None and spec.kind in APPEARANCE_FIELD_WIDGETS:
            builder = _qt_field_control
        if builder is None:
            continue
        widget = builder(spec, target.visual_ids, viewer.controller, container)
        if spec.kind not in STATIC_CONTROL_KINDS:
            viewer.controller.connect_widget(
                widget, subscription_specs=widget.subscription_specs()
            )
        if closeables is not None:
            closeables.append(widget)
        layout.addWidget(widget.widget)

    layout.addStretch()
    return container


def _render_channel_controls_qt(
    viewer: object, closeables: list | None = None
) -> object | None:
    """Build and wire a ``QtChannelList`` for the configured channel visual(s).

    Multi-scene aware: for an ``OrthoViewer`` the single widget drives every
    panel's sibling visual via the fan-out ``visual_ids``.  Returns ``None``
    when no channel controls are configured.
    """
    from cellier.convenience.layout._shared import (
        _resolve_channel_visual_ids,
        channel_widget_kwargs,
    )
    from cellier.gui.qt.visuals import QtChannelList

    resolved = _resolve_channel_visual_ids(viewer)
    if resolved is None:
        return None
    config, visual_ids, channels = resolved

    widget = QtChannelList(
        visual_ids, channels, **channel_widget_kwargs(config, channels)
    )
    viewer.controller.connect_widget(
        widget, subscription_specs=widget.subscription_specs()
    )
    if closeables is not None:
        closeables.append(widget)
    return widget.widget


def _render_render_controls_qt(
    spec: object, viewer: object, closeables: list | None = None
) -> object | None:
    """Build and wire one Qt panel per section the spec names.

    Needs no configured visual: render settings belong to the renderer, so
    unlike the appearance dock this never returns ``None`` for want of a
    target.
    """
    from PySide6 import QtWidgets
    from PySide6.QtWidgets import QSizePolicy

    from cellier.convenience.layout._shared import (
        render_panel_kwargs,
        render_panel_sections,
    )
    from cellier.gui._render_controls import RENDER_DOCK_TITLE
    from cellier.gui.qt.render import (
        QtAmbientOcclusionControls,
        QtOutlineControls,
        QtTemporalControls,
    )

    panel_types = {
        "outline": QtOutlineControls,
        "ambient_occlusion": QtAmbientOcclusionControls,
        "temporal": QtTemporalControls,
    }
    sections = render_panel_sections(spec)
    if not sections:
        return None

    controller = viewer.controller
    inner = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(inner)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)

    for section in sections:
        panel = panel_types[section](
            getattr(controller.render_config, section),
            **render_panel_kwargs(section, controller),
        )
        controller.connect_widget(panel, subscription_specs=panel.subscription_specs())
        if closeables is not None:
            closeables.append(panel)
        layout.addWidget(_titled(panel, inner))

    layout.addStretch()

    # One heading over the whole dock, naming its scope.  Without it the
    # per-visual groups on the other side of the canvas and these read as
    # the same kind of thing -- "Outline" beside "Outlines" is not a
    # distinction anyone should have to notice.
    from cellier.gui.qt.visuals._chrome import titled_group

    container = titled_group(RENDER_DOCK_TITLE, inner)
    container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
    container.setMinimumWidth(260)
    return container


def _titled(panel, parent):
    """Wrap a render panel in a group box carrying its title."""
    from cellier.gui.qt.visuals._chrome import titled_group

    return titled_group(panel.title, panel.widget, parent)


def _render_dock_qt(
    spec: object, viewer: object, closeables: list | None = None
) -> object | None:
    """Render one dock spec to a Qt widget, or return None."""
    if spec is None:
        return None

    from PySide6 import QtWidgets

    from cellier.convenience.layout._spec import (
        AppearanceControls,
        ChannelControls,
        HStack,
        RenderControls,
        VStack,
    )

    if isinstance(spec, AppearanceControls):
        return _render_appearance_controls_qt(viewer, closeables)
    if isinstance(spec, ChannelControls):
        return _render_channel_controls_qt(viewer, closeables)
    if isinstance(spec, RenderControls):
        return _render_render_controls_qt(spec, viewer, closeables)
    if isinstance(spec, (HStack, VStack)):
        widgets = [
            widget
            for widget in (
                _render_dock_qt(item, viewer, closeables) for item in spec.items
            )
            if widget is not None
        ]
        # A stack whose contents all resolved to nothing is an empty dock, and
        # an empty dock is a titled grey rectangle beside the canvas.  The
        # anywidget renderer already returned None here; returning a container
        # unconditionally was what made ``VStack([AppearanceControls()])`` on
        # an unconfigured viewer look different per toolkit.
        if not widgets:
            return None
        container = QtWidgets.QWidget()
        box = (
            QtWidgets.QHBoxLayout(container)
            if isinstance(spec, HStack)
            else QtWidgets.QVBoxLayout(container)
        )
        box.setContentsMargins(4, 4, 4, 4)
        for widget in widgets:
            box.addWidget(widget)
        return container

    from cellier.convenience.layout._shared import unsupported_dock_node

    raise unsupported_dock_node(spec)
