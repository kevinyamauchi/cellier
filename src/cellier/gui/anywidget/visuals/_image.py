"""Volume-render controls wired to the cellier v2 event bus (anywidget).

Render mode, ISO threshold, and attenuation, combined into one widget since
the latter two are mode-dependent (mirrors ``QtVolumeRenderControls``).
"""

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

_FIELDS = ("render_mode", "iso_threshold", "attenuation")


class AnywidgetVolumeRenderControls(anywidget.AnyWidget):
    """Combined render-mode, ISO-threshold, and attenuation widget.

    Mirrors ``QtVolumeRenderControls``: a mode select plus two mode-dependent
    sliders (ISO threshold visible for ``"iso"``/``"smooth_iso"``, attenuation
    visible for ``"attenuated_mip"``), sharing one UUID so a single
    subscription covers all three fields.

    Wire to the controller after construction::

        controls = AnywidgetVolumeRenderControls(visual_id, initial_render_mode="mip", ...)
        controller.connect_widget(controls, subscription_specs=controls.subscription_specs())

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``render_mode``, ``iso_threshold``, and
        ``attenuation`` fields this widget controls.
    initial_render_mode :
        Starting render mode. Default ``"mip"``.
    initial_threshold :
        Starting ISO threshold. Default ``0.2``.
    initial_attenuation :
        Starting attenuation coefficient. Default ``1.0``.
    """

    _esm = _STATIC / "volume_render_controls.js"
    _css = _STATIC / "volume_render_controls.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    render_mode = traitlets.Unicode("mip").tag(sync=True)
    iso_threshold = traitlets.Float(0.2).tag(sync=True)
    attenuation = traitlets.Float(1.0).tag(sync=True)

    def __init__(
        self,
        visual_id: UUID,
        *,
        initial_render_mode: str = "mip",
        initial_threshold: float = 0.2,
        initial_attenuation: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            render_mode=str(initial_render_mode),
            iso_threshold=float(initial_threshold),
            attenuation=float(initial_attenuation),
            **kwargs,
        )
        self._id = uuid4()
        self._visual_id = visual_id
        self._applying = False
        self.observe(self._on_trait_change, names=list(_FIELDS))

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self) -> AnywidgetVolumeRenderControls:
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
            AppearanceUpdateEvent(
                source_id=self._id,
                visual_id=self._visual_id,
                field=change["name"],
                value=change["new"],
            )
        )
