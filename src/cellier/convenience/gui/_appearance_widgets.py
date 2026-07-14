"""Shared construction of the split anywidget appearance sub-widgets.

Mirrors ``_qt_renderer.py``'s ``_render_appearance_controls_qt``: reads a
``controls_config`` + visual, and builds/wires one sub-widget per requested
appearance field (plus the always-on AABB and dataset-info widgets).  Used by
both ``convenience.layout._anywidget_renderer`` and
``convenience.gui._canvas`` so the controls_config -> widget-list mapping is
defined once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cellier.gui._colormap_util import colormap_to_str

if TYPE_CHECKING:
    from cellier.controller import CellierController
    from cellier.convenience._hosts import LayoutHost
    from cellier.convenience.gui._controls_config import BaseControlsConfig
    from cellier.visuals._base_visual import BaseVisual

_RENDER_FIELDS = {"render_mode", "iso_threshold", "attenuation"}


def build_appearance_widgets_anywidget(
    visual: BaseVisual,
    controls_config: BaseControlsConfig,
    controller: CellierController,
) -> list[object]:
    """Build and wire the anywidget appearance sub-widgets for *visual*.

    Returns the constructed, already-``connect_widget``-wired sub-widgets in
    display order (colormap, contrast limits, volume render, LOD bias, AABB,
    dataset info).  Empty when *controls_config* requests no appearance
    fields or *visual* has no ``appearance``.
    """
    from cellier.convenience.gui._controls_config import (
        InMemoryImageControlsConfig,
        MultiscaleImageControlsConfig,
    )
    from cellier.gui.anywidget import AnywidgetDatasetInfo
    from cellier.gui.anywidget.visuals import (
        AnywidgetAABBWidget,
        AnywidgetClimSlider,
        AnywidgetColormapControl,
        AnywidgetLodBiasSlider,
        AnywidgetVolumeRenderControls,
    )

    field_list = (
        controls_config.appearance
        if isinstance(controls_config.appearance, list) and controls_config.appearance
        else None
    )
    if not field_list or not hasattr(visual, "appearance"):
        return []

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

    widgets: list = []

    def _wire(w: object) -> object:
        controller.connect_widget(w, subscription_specs=w.subscription_specs())
        return w

    if "color_map" in fields and hasattr(app, "color_map"):
        widgets.append(
            _wire(
                AnywidgetColormapControl(
                    visual.id,
                    initial_colormap=colormap_to_str(
                        getattr(app, "color_map", "grays")
                    ),
                    colormap_names=colormap_names,
                )
            )
        )

    if "clim" in fields and hasattr(app, "clim"):
        widgets.append(
            _wire(
                AnywidgetClimSlider(
                    visual.id,
                    clim_range=clim_range,
                    initial_clim=raw_clim,
                )
            )
        )

    if fields & _RENDER_FIELDS and any(hasattr(app, f) for f in _RENDER_FIELDS):
        widgets.append(
            _wire(
                AnywidgetVolumeRenderControls(
                    visual.id,
                    initial_render_mode=getattr(app, "render_mode", "mip"),
                    initial_threshold=getattr(app, "iso_threshold", 0.2),
                    initial_attenuation=getattr(app, "attenuation", 1.0),
                )
            )
        )

    if "lod_bias" in fields and hasattr(app, "lod_bias"):
        widgets.append(
            _wire(
                AnywidgetLodBiasSlider(
                    visual.id,
                    initial_lod_bias=float(getattr(app, "lod_bias", 1.0)),
                )
            )
        )

    if hasattr(visual, "aabb"):
        widgets.append(
            _wire(
                AnywidgetAABBWidget(
                    visual.id,
                    initial_enabled=visual.aabb.enabled,
                    initial_line_width=visual.aabb.line_width,
                    initial_color=visual.aabb.color,
                )
            )
        )

    dataset_info_html = (
        controls_config.dataset_info
        if isinstance(controls_config, MultiscaleImageControlsConfig)
        else ""
    )
    if dataset_info_html:
        widgets.append(AnywidgetDatasetInfo(dataset_info_html))

    return widgets


_TIGHT_GAP_PX = 4
"""Spacing between grouped appearance sub-widgets, mirroring the ~6px
``setSpacing`` Qt's ``_group()`` helper uses for its one shared QVBoxLayout
(``_qt_renderer.py``) -- explicit rather than relying on the host's macro
layout default (tuned for spacing unrelated blocks like canvas/dims apart).
"""


def compose_appearance_leaf(widgets: list[object], host: LayoutHost) -> object | None:
    """Compose *widgets* into a single host leaf, or ``None`` when empty."""
    if not widgets:
        return None
    if len(widgets) == 1:
        return host.leaf(widgets[0])
    return host.stack([host.leaf(w) for w in widgets], direction="v", gap=_TIGHT_GAP_PX)


def close_appearance_widgets(widgets: list[object]) -> None:
    """Close every widget in *widgets* that supports it.

    ``AnywidgetDatasetInfo`` is a static display widget with no ``close()``.
    """
    for w in widgets:
        close = getattr(w, "close", None)
        if close is not None:
            close()
