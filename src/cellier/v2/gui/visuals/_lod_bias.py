"""LOD-bias slider wired to the cellier v2 event bus."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from psygnal import Signal

from cellier.v2.events import (
    AppearanceChangedEvent,
    AppearanceUpdateEvent,
    SubscriptionSpec,
)

if TYPE_CHECKING:
    from uuid import UUID


class QtLodBiasSlider:
    """Single-value LOD-bias slider wired to the cellier v2 bus.

    Wraps a ``superqt.QLabeledDoubleSlider`` and keeps it in sync with
    ``MultiscaleImageAppearance.lod_bias`` (or the equivalent labels field)
    via ``AppearanceChangedEvent``.

    Because changing ``lod_bias`` triggers a reslice, the ``AppearanceUpdateEvent``
    is emitted on ``sliderReleased`` rather than on every ``valueChanged`` tick,
    so only one reslice fires per drag interaction.

    Wire to the controller after construction::

        slider = QtLodBiasSlider(visual_id, initial_lod_bias=1.0)
        controller.connect_widget(slider, subscription_specs=slider.subscription_specs())

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``lod_bias`` field this widget controls.
    initial_lod_bias :
        Starting value — typically ``visual_model.appearance.lod_bias``.
    lod_range :
        ``(min, max)`` for the slider range.  Defaults to ``(0.0, 5.0)``.
    decimals :
        Number of decimal places shown in the slider label.  Default is ``2``.
    parent :
        Optional Qt parent widget.
    """

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID,
        *,
        initial_lod_bias: float = 1.0,
        lod_range: tuple[float, float] = (0.0, 5.0),
        decimals: int = 2,
        parent=None,
    ) -> None:
        from qtpy.QtCore import Qt
        from superqt import QLabeledDoubleSlider

        self._id = uuid4()
        self._visual_id = visual_id

        self._slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent)
        self._slider.setRange(*lod_range)
        self._slider.setDecimals(decimals)
        self._slider.setValue(initial_lod_bias)

        # Emit only when the user releases the handle to avoid a reslice on
        # every intermediate tick while dragging.
        self._slider.sliderReleased.connect(self._on_slider_released)

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self):
        """The Qt widget to insert into a layout."""
        return self._slider

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Return the inbound subscription this widget requires."""
        return [
            SubscriptionSpec(
                event_type=AppearanceChangedEvent,
                handler=self._on_visual_changed,
                entity_id=self._visual_id,
            )
        ]

    # ── Cellier layer: model → widget ────────────────────────────────────────

    def _on_visual_changed(self, event) -> None:
        if event.source_id == self._id:
            return
        if event.field_name != "lod_bias":
            return
        self._set_value(event.new_value)

    # ── Cellier layer: widget → model ────────────────────────────────────────

    def _on_slider_released(self) -> None:
        self.changed.emit(
            AppearanceUpdateEvent(
                source_id=self._id,
                visual_id=self._visual_id,
                field="lod_bias",
                value=self._slider.value(),
            )
        )

    # ── Qt seam: push value without re-firing signals ────────────────────────

    def _set_value(self, value: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(value)
        self._slider.blockSignals(False)
