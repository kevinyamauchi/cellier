"""AABB (axis-aligned bounding box) control wired to the cellier v2 event bus (anywidget)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.events import AABBChangedEvent, AABBUpdateEvent, SubscriptionSpec

if TYPE_CHECKING:
    from uuid import UUID

_STATIC = Path(__file__).parent / "static"

_FIELDS = ("enabled", "line_width", "color")


class AnywidgetAABBWidget(anywidget.AnyWidget):
    """Bidirectional AABB parameter controls wired to the cellier v2 bus.

    Mirrors ``QtAABBWidget``: an *enabled* checkbox, a *line_width* number
    input, and a *color* swatch, sharing one UUID so a single subscription
    covers all three fields.

    Wire to the controller after construction::

        aabb = AnywidgetAABBWidget(visual_id, initial_enabled=True, ...)
        controller.connect_widget(aabb, subscription_specs=aabb.subscription_specs())

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``aabb`` params this widget controls.
    initial_enabled :
        Starting value for the *enabled* checkbox. Default ``False``.
    initial_line_width :
        Starting value for the *line_width* input. Default ``2.0``.
    initial_color :
        Starting CSS color string for the color swatch. Default ``"#ffffff"``.
    """

    _esm = _STATIC / "aabb.js"
    _css = _STATIC / "aabb.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    enabled = traitlets.Bool(False).tag(sync=True)
    line_width = traitlets.Float(2.0).tag(sync=True)
    color = traitlets.Unicode("#ffffff").tag(sync=True)

    def __init__(
        self,
        visual_id: UUID,
        *,
        initial_enabled: bool = False,
        initial_line_width: float = 2.0,
        initial_color: str = "#ffffff",
        **kwargs,
    ) -> None:
        super().__init__(
            enabled=bool(initial_enabled),
            line_width=float(initial_line_width),
            color=str(initial_color),
            **kwargs,
        )
        self._id = uuid4()
        self._visual_id = visual_id
        self._applying = False
        self.observe(self._on_trait_change, names=list(_FIELDS))

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self) -> AnywidgetAABBWidget:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Return the inbound subscription this widget requires."""
        return [
            SubscriptionSpec(
                event_type=AABBChangedEvent,
                handler=self._on_aabb_changed,
                entity_id=self._visual_id,
            )
        ]

    # ── model -> widget ──────────────────────────────────────────────────────

    def _on_aabb_changed(self, event: AABBChangedEvent) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        if event.field_name not in _FIELDS:
            return
        self._set_field(event.field_name, event.new_value)

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
            AABBUpdateEvent(
                source_id=self._id,
                visual_id=self._visual_id,
                field=change["name"],
                value=change["new"],
            )
        )
