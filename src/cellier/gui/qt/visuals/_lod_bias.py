"""LOD-bias slider wired to the cellier v2 event bus."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from psygnal import Signal

from cellier.events import (
    AppearanceChangedEvent,
    AppearanceUpdateEvent,
    SubscriptionSpec,
)
from cellier.gui._appearance_fields import VisualIdGroup

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID


class QtLodBiasSlider(VisualIdGroup):
    """Single-value LOD-bias slider wired to the cellier v2 bus.

    Wraps a ``superqt.QLabeledDoubleSlider`` and keeps it in sync with
    ``MultiscaleImageAppearance.lod_bias`` (or the equivalent labels field)
    via ``AppearanceChangedEvent``.

    Because changing ``lod_bias`` triggers a reslice, the ``AppearanceUpdateEvent``
    is emitted on ``sliderReleased`` rather than on every ``valueChanged`` tick,
    so only one reslice fires per drag interaction.

    Wire to the controller after construction::

        slider = QtLodBiasSlider(visual_id, initial_lod_bias=1.0)
        controller.connect_widget(
            slider, subscription_specs=slider.subscription_specs()
        )

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``lod_bias`` field this widget controls.
        A sequence drives every listed visual in lock-step -- the
        ``OrthoViewer``'s four panel siblings (design section 8.1).
    initial_lod_bias :
        Starting value — typically ``visual_model.appearance.lod_bias``.
    lod_range :
        ``(min, max)`` for the slider range.  Defaults to ``(1e-6, 5.0)``.
    decimals :
        Number of decimal places shown in the slider label.  Default is ``2``.
    parent :
        Optional Qt parent widget.
    """

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial_lod_bias: float = 1.0,
        lod_range: tuple[float, float] = (1e-6, 5.0),
        decimals: int = 2,
        parent=None,
    ) -> None:
        from qtpy.QtCore import Qt
        from superqt import QLabeledDoubleSlider

        self._id = uuid4()
        self._init_visual_ids(visual_id)

        self._slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent)
        self._slider.setRange(*lod_range)
        self._slider.setValue(initial_lod_bias)
        self._slider.setDecimals(decimals)

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
        return self._group_specs(AppearanceChangedEvent, self._on_visual_changed)

    # ── Cellier layer: model → widget ────────────────────────────────────────

    def _on_visual_changed(self, event) -> None:
        if event.source_id == self._id:
            return
        if event.field_name != "lod_bias":
            return
        self._set_value(event.new_value)

    # ── Cellier layer: widget → model ────────────────────────────────────────

    def _on_slider_released(self) -> None:
        self._emit_group(AppearanceUpdateEvent, "lod_bias", self._slider.value())

    # ── Qt seam: push value without re-firing signals ────────────────────────

    def _set_value(self, value: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(value)
        self._slider.blockSignals(False)
