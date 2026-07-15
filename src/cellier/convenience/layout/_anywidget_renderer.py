"""Anywidget renderer -- the view layer for Layout specs on anywidget hosts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellier.convenience._hosts import LayoutHost
    from cellier.convenience.layout._spec import Layout


class _RenderView:
    """Rendered view with teardown tracking.

    Holds the composed root widget and every closeable object built during
    rendering.  ``close()`` tears them all down idempotently.
    """

    def __init__(self, root: object, closeables: list) -> None:
        self.root = root
        self._closeables = closeables
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for obj in self._closeables:
            try:
                obj.close()
            except Exception:
                pass


def render_anywidget(layout: Layout, viewer: object, host: LayoutHost) -> _RenderView:
    """Render a Layout spec to an anywidget host.

    Walks the spec, builds dock widgets from viewer state, and returns a
    :class:`_RenderView` whose ``.root`` is ready for ``host.present()`` and
    whose ``.close()`` tears down all created widgets.
    """
    closeables: list = []

    center = _render_center(layout.center, host, closeables)

    left = _render_dock(layout.left_dock, viewer, host, closeables)
    right = _render_dock(layout.right_dock, viewer, host, closeables)
    top = _render_dock(layout.top_dock, viewer, host, closeables)
    bottom = _render_dock(layout.bottom_dock, viewer, host, closeables)

    # Middle row: [left | center | right]
    middle_items = [w for w in (left, center, right) if w is not None]
    middle = (
        host.stack(middle_items, direction="h")
        if len(middle_items) > 1
        else middle_items[0]
    )

    # Outer column: [top / middle / bottom]
    outer_items = [w for w in (top, middle, bottom) if w is not None]
    if len(outer_items) == 1:
        root = outer_items[0]
    else:
        # No explicit align: the default cross-axis "stretch" is required for
        # the tree to fill the notebook cell / sidecar tab width; "center"
        # (the previous value) shrinks the whole tree to its content's
        # natural width and centers it, defeating the responsive-width CSS.
        root = host.stack(outer_items, direction="v")

    return _RenderView(root, closeables)


def _render_center(node: object, host: LayoutHost, closeables: list) -> object:
    """Recursively render a center spec node to a composed widget."""
    from cellier.convenience.layout._spec import Grid, HStack, VStack

    if isinstance(node, HStack):
        items = [_render_center(item, host, closeables) for item in node.items]
        return host.stack(items, direction="h")
    if isinstance(node, VStack):
        items = [_render_center(item, host, closeables) for item in node.items]
        return host.stack(items, direction="v")
    if isinstance(node, Grid):
        rows = [
            [_render_center(cell, host, closeables) for cell in row if cell is not None]
            for row in node.cells
        ]
        return host.grid(rows)
    # Leaf: AnywidgetCanvasView, OrthoAnywidgetCanvases, or any object with compose().
    if hasattr(node, "close"):
        closeables.append(node)
    return node.compose(host)


def _render_dock(
    spec: object,
    viewer: object,
    host: LayoutHost,
    closeables: list,
) -> object | None:
    """Render one dock spec to a widget, or return None if the dock is empty."""
    if spec is None:
        return None

    from cellier.convenience.layout._spec import (
        AppearanceControls,
        ChannelControls,
        HStack,
        VStack,
    )

    if isinstance(spec, AppearanceControls):
        return _render_appearance_controls(viewer, host, closeables)
    if isinstance(spec, ChannelControls):
        return _render_channel_controls(viewer, host, closeables)
    if isinstance(spec, (HStack, VStack)):
        direction = "h" if isinstance(spec, HStack) else "v"
        items = [_render_dock(item, viewer, host, closeables) for item in spec.items]
        items = [i for i in items if i is not None]
        return host.stack(items, direction=direction) if items else None
    return None


def _render_appearance_controls(
    viewer: object,
    host: LayoutHost,
    closeables: list,
) -> object | None:
    """Build and wire the appearance sub-widgets for the first configured visual.

    Mirrors ``_render_appearance_controls_qt``: builds one anywidget per
    requested appearance field via ``build_appearance_widgets_anywidget``,
    then composes them into a single leaf for this dock slot.
    """
    from cellier.convenience.gui._appearance_widgets import (
        build_appearance_widgets_anywidget,
        compose_appearance_leaf,
    )
    from cellier.convenience.gui._controls_config import ChannelControlsConfig

    controls_configs: dict = getattr(viewer, "_controls_configs", {})
    scene = getattr(viewer, "scene", None)
    if scene is None or not controls_configs:
        return None

    controls_config = None
    visual = None
    for v in scene.visuals:
        cfg = controls_configs.get(v.id)
        if cfg is not None and not isinstance(cfg, ChannelControlsConfig):
            visual = v
            controls_config = cfg
            break

    if controls_config is None:
        return None

    widgets = build_appearance_widgets_anywidget(
        visual, controls_config, viewer.controller
    )
    if not widgets:
        return None

    closeables.extend(widgets)
    return compose_appearance_leaf(widgets, host)


def _render_channel_controls(
    viewer: object,
    host: LayoutHost,
    closeables: list,
) -> object | None:
    """Build and wire an ``AnywidgetChannelList`` for the configured channel visual(s).

    Multi-scene aware: for an ``OrthoViewer`` the single widget drives every
    panel's sibling visual via the fan-out ``visual_ids``.  Returns ``None``
    when no channel controls are configured.
    """
    from cellier.convenience.layout._shared import (
        _resolve_channel_visual_ids,
        channel_widget_kwargs,
    )
    from cellier.gui.anywidget.visuals import AnywidgetChannelList

    resolved = _resolve_channel_visual_ids(viewer)
    if resolved is None:
        return None
    config, visual_ids, channels = resolved

    widget = AnywidgetChannelList(
        visual_ids, channels, **channel_widget_kwargs(config, channels)
    )
    viewer.controller.connect_widget(
        widget, subscription_specs=widget.subscription_specs()
    )
    closeables.append(widget)
    return host.leaf(widget)
