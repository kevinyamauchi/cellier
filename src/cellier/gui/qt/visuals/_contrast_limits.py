"""Contrast-limits range slider wired to the cellier v2 event bus."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from psygnal import Signal

from cellier.events import (
    AppearanceChangedEvent,
    AppearanceUpdateEvent,
    SubscriptionSpec,
)

if TYPE_CHECKING:
    from uuid import UUID


class QtClimRangeSlider:
    """Bidirectional contrast-limits slider wired to the cellier v2 bus.

    Wraps a ``superqt.QLabeledDoubleRangeSlider`` and keeps it in sync with
    ``MultiscaleImageAppearance.clim`` via ``AppearanceChangedEvent``.  Follows the v2
    widget pattern: one UUID per widget, source-ID echo filtering, and signal
    blocking when applying model-driven updates.

    Wire to the controller after construction::

        slider = QtClimRangeSlider(visual_id, clim_range=(0, 255), initial_clim=(0, 200))
        controller.connect_widget(slider, subscription_specs=slider.subscription_specs())

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``clim`` field this widget controls.
    clim_range :
        ``(min, max)`` for the slider range.
    initial_clim :
        Starting value — typically ``visual_model.appearance.clim``.
    decimals :
        Number of decimal places shown in the slider label.  Use ``0`` for
        integer dtypes and ``2`` (or similar) for float data.  Default is ``2``.
    parent :
        Optional Qt parent widget.
    """

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID,
        *,
        clim_range: tuple[float, float],
        initial_clim: tuple[float, float],
        decimals: int = 2,
        parent=None,
    ) -> None:
        from qtpy.QtCore import Qt
        from superqt import QLabeledDoubleRangeSlider

        # ── Cellier layer ────────────────────────────────────────────────────
        self._id = uuid4()
        self._visual_id = visual_id

        # ── Qt seam 1: widget creation and signal wiring ─────────────────────
        from qtpy.QtGui import QFontMetrics

        self._slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal, parent)
        self._slider.setRange(*clim_range)
        self._slider.setValue(initial_clim)  # creates _handle_labels lazily
        self._slider.setDecimals(decimals)

        # QLabeledDoubleRangeSlider.SliderLabel._get_size() sizes from str(float)
        # repr ("0.0") but setDecimals(2) displays "0.00" -- one char wider,
        # causing clipping. LabelIsRange edge labels use 7-digit sizing (~70px
        # each) which collapses the slider track. Fix both by patching _update_size
        # to a no-op and forcing correct widths from font metrics.
        _fm = QFontMetrics(self._slider._min_label.font())
        _lw = (
            max(
                _fm.horizontalAdvance(f"{clim_range[0]:.{decimals}f}"),
                _fm.horizontalAdvance(f"{clim_range[1]:.{decimals}f}"),
            )
            + 12
        )
        self._slider._min_label._update_size = lambda *_: None
        self._slider._max_label._update_size = lambda *_: None
        self._slider._min_label.setFixedWidth(_lw)
        self._slider._max_label.setFixedWidth(_lw)
        for _hl in self._slider._handle_labels:
            _hl._update_size = lambda *_: None
            _hl.setFixedWidth(_lw)
        self._slider.valueChanged.connect(self._on_slider_changed)

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self):
        """The Qt widget to insert into a layout.

        Qt seam 1: replace with the backend element for other toolkits.
        """
        return self._slider

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Return the inbound subscription this widget requires.

        Pass the result to ``CellierController.connect_widget``.
        """
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
            return  # echo from our own change; ignore
        if event.field_name != "clim":
            return  # a different appearance field changed; nothing to do
        self._set_value(event.new_value)

    # ── Cellier layer: widget → model ────────────────────────────────────────

    def _on_slider_changed(self, value: tuple[float, float]) -> None:
        self.changed.emit(
            AppearanceUpdateEvent(
                source_id=self._id,
                visual_id=self._visual_id,
                field="clim",
                value=value,
            )
        )

    # ── Qt seam 2: push value without re-firing valueChanged ─────────────────

    def _set_value(self, value: tuple[float, float]) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(value)
        self._slider.blockSignals(False)
