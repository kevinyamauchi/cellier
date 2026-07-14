"""LOD-bias slider wired to the cellier v2 event bus (anywidget)."""

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

if TYPE_CHECKING:
    from uuid import UUID

_STATIC = Path(__file__).parent / "static"


class AnywidgetLodBiasSlider(anywidget.AnyWidget):
    """Single-value LOD-bias slider wired to the cellier v2 bus.

    Mirrors ``QtLodBiasSlider``.  Because changing ``lod_bias`` triggers a
    reslice, the JS emits only on settled ``change`` (not on every drag
    ``input``), so only one reslice fires per drag interaction.

    Wire to the controller after construction::

        slider = AnywidgetLodBiasSlider(visual_id, initial_lod_bias=1.0)
        controller.connect_widget(slider, subscription_specs=slider.subscription_specs())

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``lod_bias`` field this widget controls.
    initial_lod_bias :
        Starting value -- typically ``visual_model.appearance.lod_bias``.
    """

    _esm = _STATIC / "lod_bias.js"
    _css = _STATIC / "lod_bias.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    lod_bias = traitlets.Float(1.0).tag(sync=True)

    def __init__(
        self,
        visual_id: UUID,
        *,
        initial_lod_bias: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(lod_bias=float(initial_lod_bias), **kwargs)
        self._id = uuid4()
        self._visual_id = visual_id
        self._applying = False
        self.observe(self._on_trait_change, names="lod_bias")

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self) -> AnywidgetLodBiasSlider:
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
        if event.field_name != "lod_bias":
            return
        self._set_field("lod_bias", event.new_value)

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
                field="lod_bias",
                value=change["new"],
            )
        )
