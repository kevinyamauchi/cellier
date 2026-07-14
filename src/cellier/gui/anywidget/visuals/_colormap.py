"""Colormap control wired to the cellier v2 event bus (anywidget)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.events import (
    AppearanceChangedEvent,
    AppearanceUpdateEvent,
    SubscriptionSpec,
)
from cellier.gui._colormap_util import colormap_to_str

if TYPE_CHECKING:
    from uuid import UUID

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


class AnywidgetColormapControl(anywidget.AnyWidget):
    """Bidirectional colormap selector wired to the cellier v2 bus.

    Mirrors ``QtColormapComboBox``: one UUID per widget, source-ID echo
    filtering, and a narrow subscription to just the ``color_map`` field.

    Wire to the controller after construction::

        control = AnywidgetColormapControl(visual_id, initial_colormap="grays")
        controller.connect_widget(control, subscription_specs=control.subscription_specs())

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``color_map`` field this widget controls.
    initial_colormap :
        Starting colormap -- typically ``visual_model.appearance.color_map``.
    colormap_names :
        Available colormap names for the dropdown.  Defaults to a curated list.
    """

    _esm = _STATIC / "colormap_control.js"
    _css = _STATIC / "colormap_control.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    color_map = traitlets.Unicode("grays").tag(sync=True)
    colormap_names = traitlets.List([]).tag(sync=True)

    def __init__(
        self,
        visual_id: UUID,
        *,
        initial_colormap: str = "grays",
        colormap_names: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            color_map=colormap_to_str(initial_colormap),
            colormap_names=list(colormap_names)
            if colormap_names is not None
            else list(_DEFAULT_COLORMAP_NAMES),
            **kwargs,
        )
        self._id = uuid4()
        self._visual_id = visual_id
        self._applying = False
        self.observe(self._on_trait_change, names="color_map")

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self) -> AnywidgetColormapControl:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Return the inbound subscription this widget requires."""
        return [
            SubscriptionSpec(
                event_type=AppearanceChangedEvent,
                handler=self._on_appearance_changed,
                entity_id=self._visual_id,
            )
        ]

    # ── model -> widget ──────────────────────────────────────────────────────

    def _on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        if event.field_name != "color_map":
            return
        self._set_field("color_map", colormap_to_str(event.new_value))

    def _set_field(self, name: str, value) -> None:
        self._applying = True
        try:
            setattr(self, name, value)
        finally:
            self._applying = False

    # ── widget -> model ──────────────────────────────────────────────────────

    def _on_trait_change(self, change) -> None:
        if self._applying:
            return
        self.changed.emit(
            AppearanceUpdateEvent(
                source_id=self._id,
                visual_id=self._visual_id,
                field="color_map",
                value=change["new"],
            )
        )
