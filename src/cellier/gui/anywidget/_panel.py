"""Composite anywidget control panel (Design A).

A single ``ControlPanel`` renders every control as raw DOM inside one ESM,
synced through traitlets.  Phase 1 implements the dims sliders; Phase 2 adds
appearance controls (render mode, ISO threshold, attenuation, LOD bias,
contrast limits, colormap, AABB, and dataset info) as additional traits + DOM
rows (see the design doc sections 6.3 / 6.4 and the Phase 2 list in the plan).

The panel satisfies the :class:`cellier.gui._protocol.WidgetView` contract: it
carries one ``_id``, a ``changed`` / ``closed`` psygnal pair, a ``widget``
property, and ``subscription_specs()`` -- exactly like the Qt widgets, so
``CellierController.connect_widget`` wires it the same way.

Notes on the traitlet <-> JSON boundary
----------------------------------------
JSON object keys are always strings, so the synced dict traits (``slice_indices``,
``axis_ranges``, ``axis_labels``) use ``str(axis_index)`` keys.  The cellier
layer converts to / from ``int`` axis indices at the event boundary
(``_on_dims_changed`` in, ``_emit_dims`` out).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.events import (
    AABBChangedEvent,
    AABBUpdateEvent,
    AppearanceChangedEvent,
    AppearanceUpdateEvent,
    SubscriptionSpec,
)
from cellier.gui._colormap_util import colormap_to_str as _colormap_to_str

if TYPE_CHECKING:
    from uuid import UUID

    from cellier.scene.scene import Scene

_STATIC = Path(__file__).parent / "static"

_DEFAULT_COLORMAP_NAMES = [
    "grays",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "turbo",
    "hot",
    "cool",
    "bwr",
]

# Appearance fields that ControlPanel handles (order determines JS display order).
_APPEARANCE_FIELDS = [
    "color_map",
    "clim",
    "render_mode",
    "iso_threshold",
    "attenuation",
    "lod_bias",
]

# AABB trait name -> model field name
_AABB_TRAIT_TO_FIELD: dict[str, str] = {
    "aabb_enabled": "enabled",
    "aabb_line_width": "line_width",
    "aabb_color": "color",
}


class ControlPanel(anywidget.AnyWidget):
    """Composite raw-DOM control panel for the anywidget front-end.

    Phase 1 exposes the per-axis dims sliders.  Phase 2 adds appearance
    controls (render mode, ISO threshold, attenuation, LOD bias, contrast
    limits, colormap, AABB, and an optional dataset-info detail block).

    Construct via :meth:`from_scene` (dims only) or by passing ``visual_id``
    and related keyword arguments for full appearance support.  Wire with::

        panel = ControlPanel.from_scene(scene, axis_ranges, visual=visual)
        controller.connect_widget(panel, subscription_specs=panel.subscription_specs())

    Parameters
    ----------
    scene_id : UUID
        UUID of the scene whose slice indices this panel controls.
    axis_ranges : dict
        Mapping of ``str(axis)`` to ``[min, max]`` slider bounds.
    axis_labels : dict
        Mapping of ``str(axis)`` to display label.
    slice_indices : dict
        Mapping of ``str(axis)`` to the current integer slice position.
    displayed_axes : list
        Axis indices currently displayed (their rows are hidden).
    stacked_axes : list
        Axis indices composited by the render layer (their rows are hidden).
    non_displayed : list
        Axis indices never shown as sliders regardless of dims state.
    visual_id : UUID or None
        UUID of the visual whose appearance this panel controls.  When
        ``None``, the appearance section is hidden.
    appearance_fields : list[str] or None
        Which appearance field names this visual actually supports.  Controls
        which rows are rendered in JS.  Defaults to none (no appearance rows).
    render_mode : str
        Initial render mode.  Default ``"mip"``.
    iso_threshold : float
        Initial ISO threshold.  Default ``0.2``.
    attenuation : float
        Initial attenuation coefficient.  Default ``1.0``.
    lod_bias : float
        Initial LOD bias.  Default ``1.0``.
    clim : tuple or list
        Initial contrast limits ``[lo, hi]``.  Default ``[0.0, 1.0]``.
    clim_range : tuple or list
        Slider bounds for the contrast limits ``[min, max]``.
        Default ``[0.0, 1.0]``.
    color_map : str
        Initial colormap name.  Default ``"grays"``.
    colormap_names : list[str] or None
        Available colormap names for the dropdown.  Defaults to a curated list.
    aabb_enabled : bool
        Initial bounding-box visibility.  Default ``False``.
    aabb_line_width : float
        Initial AABB line width in pixels.  Default ``2.0``.
    aabb_color : str
        Initial AABB color as a CSS hex string.  Default ``"#ffffff"``.
    dataset_info : str
        Pre-formatted HTML for the dataset-info detail block.  Empty string
        hides the block.
    """

    _esm = _STATIC / "panel.js"
    _css = _STATIC / "panel.css"

    # psygnal outward signals (the WidgetView contract); not traitlets.
    changed: Signal = Signal(object)
    closed: Signal = Signal()

    # ── Appearance traits (Phase 2) ──────────────────────────────────────────
    has_appearance = traitlets.Bool(False).tag(sync=True)
    appearance_fields = traitlets.List([]).tag(sync=True)
    render_mode = traitlets.Unicode("mip").tag(sync=True)
    iso_threshold = traitlets.Float(0.2).tag(sync=True)
    attenuation = traitlets.Float(1.0).tag(sync=True)
    lod_bias = traitlets.Float(1.0).tag(sync=True)
    clim = traitlets.List([0.0, 1.0]).tag(sync=True)
    clim_range = traitlets.List([0.0, 1.0]).tag(sync=True)
    color_map = traitlets.Unicode("grays").tag(sync=True)
    colormap_names = traitlets.List([]).tag(sync=True)

    # ── AABB traits (Phase 2) ────────────────────────────────────────────────
    aabb_enabled = traitlets.Bool(False).tag(sync=True)
    aabb_line_width = traitlets.Float(2.0).tag(sync=True)
    aabb_color = traitlets.Unicode("#ffffff").tag(sync=True)

    # ── Dataset info (Phase 2, static) ──────────────────────────────────────
    dataset_info = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        *,
        visual_id: UUID | None = None,
        appearance_fields: list[str] | None = None,
        render_mode: str = "mip",
        iso_threshold: float = 0.2,
        attenuation: float = 1.0,
        lod_bias: float = 1.0,
        clim: tuple[float, float] | list = (0.0, 1.0),
        clim_range: tuple[float, float] | list = (0.0, 1.0),
        color_map: str = "grays",
        colormap_names: list[str] | None = None,
        aabb_enabled: bool = False,
        aabb_line_width: float = 2.0,
        aabb_color: str = "#ffffff",
        dataset_info: str = "",
        **kwargs,
    ) -> None:
        has_visual = visual_id is not None
        resolved_fields = list(appearance_fields or [])
        super().__init__(
            has_appearance=has_visual,
            appearance_fields=resolved_fields,
            render_mode=str(render_mode),
            iso_threshold=float(iso_threshold),
            attenuation=float(attenuation),
            lod_bias=float(lod_bias),
            clim=[float(clim[0]), float(clim[1])],
            clim_range=[float(clim_range[0]), float(clim_range[1])],
            color_map=str(color_map),
            colormap_names=colormap_names
            if colormap_names is not None
            else _DEFAULT_COLORMAP_NAMES,
            aabb_enabled=bool(aabb_enabled),
            aabb_line_width=float(aabb_line_width),
            aabb_color=str(aabb_color),
            dataset_info=str(dataset_info),
            **kwargs,
        )
        self._id = uuid4()
        self._visual_id: UUID | None = visual_id
        self._appearance_fields: frozenset[str] = frozenset(resolved_fields)
        self._applying = False

        # Appearance observers — fire on JS save_changes writes
        self.observe(
            self._on_appearance_trait,
            names=["render_mode", "iso_threshold", "attenuation", "color_map"],
        )
        self.observe(self._on_lod_bias_trait, names=["lod_bias"])
        self.observe(self._on_clim_trait, names=["clim"])

        # AABB observers
        self.observe(
            self._on_aabb_trait,
            names=["aabb_enabled", "aabb_line_width", "aabb_color"],
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_scene(
        cls,
        scene: Scene,
        axis_ranges: dict[int, tuple[float, float]],
        *,
        non_displayed: tuple[int, ...] = (),
        visual=None,
        appearance_fields: list[str] | None = None,
        colormap_names: list[str] | None = None,
        clim_range: tuple[float, float] | None = None,
        dataset_info: str = "",
    ) -> ControlPanel:
        """Build a panel from a live scene.

        Derives axis labels and the initial dims state from *scene*; the caller
        supplies *axis_ranges* (which needs data-store knowledge not on the
        dims model itself).

        When *visual* is provided, the appearance section is enabled and
        populated from the visual's current ``appearance`` and ``aabb`` fields.

        Parameters
        ----------
        scene : Scene
            The scene whose dims this panel controls.
        axis_ranges : dict[int, tuple[float, float]]
            Mapping of axis index to ``(world_min, world_max)``.
        non_displayed : tuple[int, ...]
            Axes excluded from the sliders regardless of dims state.
        visual : BaseVisual or None
            Visual model to extract appearance data from.  When ``None``,
            only the dims section is shown.
        appearance_fields : list[str] or None
            Ordered list of appearance field names to display.  When
            provided, this order is used directly (filtered by the fields
            the visual's appearance actually has).  When ``None``, the
            fixed ``_APPEARANCE_FIELDS`` order is used as a fallback.
        colormap_names : list[str] or None
            Available colormap names for the dropdown.  Defaults to a curated
            list when ``None``.
        clim_range : tuple[float, float] or None
            ``(min, max)`` bounds for the contrast-limits slider.  Inferred
            from the visual's current clim when ``None``.
        dataset_info : str
            Pre-formatted HTML for the dataset info block.  Empty hides it.

        Returns
        -------
        ControlPanel
        """
        kwargs: dict = {
            "colormap_names": colormap_names,
            "dataset_info": dataset_info,
        }

        if visual is not None and hasattr(visual, "appearance"):
            app = visual.appearance
            field_order = (
                appearance_fields
                if appearance_fields is not None
                else _APPEARANCE_FIELDS
            )
            avail = [f for f in field_order if hasattr(app, f)]
            raw_clim = list(getattr(app, "clim", (0.0, 1.0)))
            resolved_clim_range = (
                [float(clim_range[0]), float(clim_range[1])]
                if clim_range is not None
                else [min(0.0, raw_clim[0]), max(1.0, raw_clim[1])]
            )
            kwargs.update(
                visual_id=visual.id,
                appearance_fields=avail,
                render_mode=getattr(app, "render_mode", "mip"),
                iso_threshold=getattr(app, "iso_threshold", 0.2),
                attenuation=getattr(app, "attenuation", 1.0),
                lod_bias=getattr(app, "lod_bias", 1.0),
                clim=raw_clim,
                clim_range=resolved_clim_range,
                color_map=_colormap_to_str(getattr(app, "color_map", "grays")),
                aabb_enabled=visual.aabb.enabled,
                aabb_line_width=visual.aabb.line_width,
                aabb_color=visual.aabb.color,
            )

        return cls(**kwargs)

    # ------------------------------------------------------------------
    # WidgetView contract
    # ------------------------------------------------------------------

    @property
    def widget(self) -> ControlPanel:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Return the inbound subscriptions this panel requires."""
        specs: list[SubscriptionSpec] = []
        if self._visual_id is not None:
            specs.append(
                SubscriptionSpec(
                    event_type=AppearanceChangedEvent,
                    handler=self._on_appearance_changed,
                    entity_id=self._visual_id,
                )
            )
            specs.append(
                SubscriptionSpec(
                    event_type=AABBChangedEvent,
                    handler=self._on_aabb_changed,
                    entity_id=self._visual_id,
                )
            )
        return specs

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    # ------------------------------------------------------------------
    # model -> widget: appearance (Phase 2)
    # ------------------------------------------------------------------

    def _on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        field = event.field_name
        if field not in self._appearance_fields:
            return
        value = event.new_value
        if field == "color_map":
            value = _colormap_to_str(value)
        elif field == "clim":
            value = [float(value[0]), float(value[1])]
        self._set_field(field, value)

    def _on_aabb_changed(self, event: AABBChangedEvent) -> None:
        if event.source_id == self._id:
            return
        trait_map = {v: k for k, v in _AABB_TRAIT_TO_FIELD.items()}
        trait = trait_map.get(event.field_name)
        if trait is None:
            return
        self._set_field(trait, event.new_value)

    def _set_field(self, name: str, value) -> None:
        """Set a synced trait without echoing it back onto the bus."""
        self._applying = True
        try:
            setattr(self, name, value)
        finally:
            self._applying = False

    # ------------------------------------------------------------------
    # widget -> model: appearance (Phase 2)
    # ------------------------------------------------------------------

    def _on_appearance_trait(self, change) -> None:
        if self._applying or self._visual_id is None:
            return
        field = change["name"]
        if field not in self._appearance_fields:
            return
        self.changed.emit(
            AppearanceUpdateEvent(
                source_id=self._id,
                visual_id=self._visual_id,
                field=field,
                value=change["new"],
            )
        )

    def _on_lod_bias_trait(self, change) -> None:
        # lod_bias triggers a reslice; JS emits only on settled change, not input.
        if self._applying or self._visual_id is None:
            return
        if "lod_bias" not in self._appearance_fields:
            return
        self.changed.emit(
            AppearanceUpdateEvent(
                source_id=self._id,
                visual_id=self._visual_id,
                field="lod_bias",
                value=change["new"],
            )
        )

    def _on_clim_trait(self, change) -> None:
        if self._applying or self._visual_id is None:
            return
        if "clim" not in self._appearance_fields:
            return
        self.changed.emit(
            AppearanceUpdateEvent(
                source_id=self._id,
                visual_id=self._visual_id,
                field="clim",
                value=tuple(change["new"]),
            )
        )

    def _on_aabb_trait(self, change) -> None:
        if self._applying or self._visual_id is None:
            return
        aabb_field = _AABB_TRAIT_TO_FIELD.get(change["name"])
        if aabb_field is None:
            return
        self.changed.emit(
            AABBUpdateEvent(
                source_id=self._id,
                visual_id=self._visual_id,
                field=aabb_field,
                value=change["new"],
            )
        )
