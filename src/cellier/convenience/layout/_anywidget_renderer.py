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
        # The leaves first, so each control emits ``closed`` and the controller
        # drops its subscriptions, then the root -- which closes the container
        # widgets the host built to compose them.  Those are widgets too, and
        # nothing else holds them, so skipping the root leaves the whole
        # scaffolding registered with ``ipywidgets``.
        for obj in [*self._closeables, self.root]:
            close = getattr(obj, "close", None)
            if close is None:
                continue
            try:
                close()
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


def _render_render_controls(
    spec: object,
    viewer: object,
    host: LayoutHost,
    closeables: list,
) -> object | None:
    """Build and wire one notebook panel per section the spec names.

    Mirrors ``_render_render_controls_qt``: both read the sections from the
    same spec and hand the panels the same derived-state callables, so the
    two front ends cannot drift.
    """
    from cellier.convenience.layout._shared import (
        render_panel_kwargs,
        render_panel_sections,
    )
    from cellier.gui.anywidget.render import (
        AnywidgetAmbientOcclusionControls,
        AnywidgetOutlineControls,
        AnywidgetTemporalControls,
    )

    panel_types = {
        "outline": AnywidgetOutlineControls,
        "ambient_occlusion": AnywidgetAmbientOcclusionControls,
        "temporal": AnywidgetTemporalControls,
    }
    sections = render_panel_sections(spec)
    if not sections:
        return None

    controller = viewer.controller
    panels = []
    for section in sections:
        panel = panel_types[section](
            getattr(controller.render_config, section),
            **render_panel_kwargs(section, controller),
        )
        controller.connect_widget(panel, subscription_specs=panel.subscription_specs())
        closeables.append(panel)
        panels.append(host.leaf(panel))

    return panels[0] if len(panels) == 1 else host.stack(panels, direction="v")


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
        RenderControls,
        VStack,
    )

    if isinstance(spec, AppearanceControls):
        return _render_appearance_controls(viewer, host, closeables)
    if isinstance(spec, ChannelControls):
        return _render_channel_controls(viewer, host, closeables)
    if isinstance(spec, RenderControls):
        return _render_render_controls(spec, viewer, host, closeables)
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

    Mirrors ``_render_appearance_controls_qt``: both resolve the target with
    ``select_appearance_target`` and build from the same ``appearance_specs``,
    so the two front ends cannot drift (design section 4.2).
    """
    from cellier.convenience.gui._appearance_widgets import (
        build_appearance_widgets_anywidget,
        compose_appearance_leaf,
    )
    from cellier.convenience.layout._shared import select_appearance_target

    target = select_appearance_target(viewer)
    if target is None:
        return None

    widgets = build_appearance_widgets_anywidget(
        target.visual, target.config, viewer.controller, target.visual_ids
    )
    if not widgets:
        return None

    closeables.extend(w for w in widgets if hasattr(w, "close"))
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
