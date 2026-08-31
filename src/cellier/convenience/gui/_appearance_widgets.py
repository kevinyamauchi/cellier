"""Anywidget view layer for the shared appearance-control specs.

The decision of *which* controls a panel contains, in what order, seeded with
what values, is made once in ``convenience.layout._shared.appearance_specs``
and shared with the Qt renderer (design section 7.3).  This module is only the
anywidget half of the view: a dispatch table from ``ControlSpec.kind`` to a
widget class, and the composition of the built widgets into one host leaf.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellier.controller import CellierController
    from cellier.convenience._hosts import LayoutHost
    from cellier.convenience.gui._controls_config import BaseControlsConfig
    from cellier.convenience.layout._shared import ControlSpec
    from cellier.visuals._base_visual import BaseVisual


def _any_color_map(spec: ControlSpec, visual_ids):
    from cellier.gui.anywidget.visuals import AnywidgetColormapControl

    return AnywidgetColormapControl(
        visual_ids,
        initial_colormap=spec.values["initial_colormap"],
        colormap_names=spec.values["colormap_names"],
    )


def _any_clim(spec: ControlSpec, visual_ids):
    from cellier.gui.anywidget.visuals import AnywidgetClimSlider

    return AnywidgetClimSlider(
        visual_ids,
        clim_range=spec.values["clim_range"],
        initial_clim=spec.values["initial_clim"],
    )


def _any_render(spec: ControlSpec, visual_ids):
    from cellier.gui.anywidget.visuals import AnywidgetVolumeRenderControls

    # No ``dtype_max`` here: the anywidget control derives its own slider
    # bounds.  Its Qt counterpart takes one, which is why ``clim_range`` is on
    # the shared spec and the keyword is not.
    return AnywidgetVolumeRenderControls(
        visual_ids,
        initial_render_mode=spec.values["initial_render_mode"],
        initial_threshold=spec.values["initial_threshold"],
        initial_attenuation=spec.values["initial_attenuation"],
    )


def _any_lod_bias(spec: ControlSpec, visual_ids):
    from cellier.gui.anywidget.visuals import AnywidgetLodBiasSlider

    return AnywidgetLodBiasSlider(
        visual_ids, initial_lod_bias=spec.values["initial_lod_bias"]
    )


def _any_aabb(spec: ControlSpec, visual_ids):
    from cellier.gui.anywidget.visuals import AnywidgetAABBWidget

    return AnywidgetAABBWidget(
        visual_ids,
        initial_enabled=spec.values["initial_enabled"],
        initial_line_width=spec.values["initial_line_width"],
        initial_color=spec.values["initial_color"],
    )


def _any_dataset_info(spec: ControlSpec, visual_ids):
    from cellier.gui.anywidget import AnywidgetDatasetInfo

    return AnywidgetDatasetInfo(spec.values["html"])


def _any_field_control(spec: ControlSpec, visual_ids):
    """Build any of the 22 single-field controls from the shared table.

    The anywidget twin of ``_qt_field_control``; see it for why one builder
    serves them all.
    """
    from cellier.gui._appearance_fields import field_widget_class

    widget_class = field_widget_class(spec.kind, "anywidget")
    kwargs = {"initial_value": spec.values["initial_value"]}
    if "choices" in spec.values:
        kwargs["choices"] = spec.values["choices"]
    return widget_class(visual_ids, **kwargs)


_ANYWIDGET_BUILDERS = {
    "color_map": _any_color_map,
    "clim": _any_clim,
    "render": _any_render,
    "lod_bias": _any_lod_bias,
    "aabb": _any_aabb,
    "dataset_info": _any_dataset_info,
}
"""``ControlSpec.kind`` -> anywidget widget constructor."""

# Static display widgets have nothing on the bus to wire or tear down.
_UNWIRED_KINDS = frozenset({"dataset_info"})


def build_appearance_widgets_anywidget(
    visual: BaseVisual,
    controls_config: BaseControlsConfig,
    controller: CellierController,
    visual_ids: list | None = None,
) -> list[tuple[str, object]]:
    """Build and wire the anywidget appearance sub-widgets for *visual*.

    Returns ``(title, widget)`` pairs in display order, each widget already
    ``connect_widget``-wired where it has a bus contract.  The title comes
    from the shared spec and is carried as data: the anywidget front end does
    not display it today (design section 6.5.1 decision 2 deferred the
    ``LayoutHost`` seam that would), but Qt does, and having one source for
    both is the point.

    *visual_ids* is every visual the controls should write to: one on a
    ``Viewer``, the four panel siblings on an ``OrthoViewer``.  Defaults to
    ``visual`` alone.

    Empty when *controls_config* requests no appearance fields or *visual* has
    no ``appearance``.
    """
    from cellier.convenience.layout._shared import (
        appearance_specs,
        warn_skipped_appearance_fields,
    )
    from cellier.gui._appearance_fields import APPEARANCE_FIELD_WIDGETS

    specs, skipped = appearance_specs(visual, controls_config)
    warn_skipped_appearance_fields(skipped, visual, controls_config)
    ids = [visual.id] if visual_ids is None else list(visual_ids)

    built: list[tuple[str, object]] = []
    for spec in specs:
        builder = _ANYWIDGET_BUILDERS.get(spec.kind)
        if builder is None and spec.kind in APPEARANCE_FIELD_WIDGETS:
            builder = _any_field_control
        if builder is None:
            continue
        widget = builder(spec, ids)
        if spec.kind not in _UNWIRED_KINDS:
            controller.connect_widget(
                widget, subscription_specs=widget.subscription_specs()
            )
        built.append((spec.title, widget))
    return built


_TIGHT_GAP_PX = 4
"""Spacing between grouped appearance sub-widgets, mirroring the ~6px
``setSpacing`` Qt's ``_group()`` helper uses for its one shared QVBoxLayout
(``_qt_renderer.py``) -- explicit rather than relying on the host's macro
layout default (tuned for spacing unrelated blocks like canvas/dims apart).
"""


def compose_appearance_leaf(
    widgets: list[tuple[str, object]], host: LayoutHost
) -> object | None:
    """Compose built ``(title, widget)`` pairs into one host leaf, or ``None``.

    Titles are dropped here rather than rendered; see
    :func:`build_appearance_widgets_anywidget`.
    """
    if not widgets:
        return None
    if len(widgets) == 1:
        return host.leaf(widgets[0][1])
    return host.stack(
        [host.leaf(w) for _title, w in widgets], direction="v", gap=_TIGHT_GAP_PX
    )
