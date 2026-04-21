"""Contrast-limits range slider wired to the cellier v2 event bus."""

from __future__ import annotations

from uuid import uuid4

from cellier.v2.events import AppearanceUpdateEvent


class QtClimRangeSlider:
    """Bidirectional contrast-limits slider wired to the cellier v2 bus.

    Wraps a ``superqt.QLabeledDoubleRangeSlider`` and keeps it in sync with
    ``ImageAppearance.clim`` via ``AppearanceChangedEvent``.  Follows the v2
    widget pattern: one UUID per widget, source-ID echo filtering, and signal
    blocking when applying model-driven updates.

    Parameters
    ----------
    controller :
        The ``CellierController`` instance.
    visual_id :
        UUID of the visual whose ``clim`` field this widget controls.
    clim_range :
        ``(min, max)`` for the slider track — typically ``(0.0, dtype_max)``.
    initial_clim :
        Starting ``(lo, hi)`` selection — typically
        ``visual_model.appearance.clim``.
    decimals :
        Number of decimal places shown in the slider labels.  Use ``0`` for
        integer dtypes and ``2`` (or similar) for float data.  Default is
        ``2``.
    parent :
        Optional Qt parent widget.
    """

    def __init__(
        self,
        controller,
        visual_id,
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
        self._controller = controller
        self._visual_id = visual_id

        # ── Qt seam 1: widget creation and signal wiring ─────────────────────
        self._slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal, parent)
        self._slider.setRange(*clim_range)
        self._slider.setValue(initial_clim)
        self._slider.setDecimals(decimals)
        self._slider.valueChanged.connect(self._on_slider_changed)

        controller.on_visual_changed(
            visual_id, self._on_visual_changed, owner_id=self._id
        )

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self):
        """The Qt widget to insert into a layout.

        Qt seam 1: replace with the backend element for other toolkits.
        """
        return self._slider

    def close(self) -> None:
        """Unsubscribe from the bus.  Call when the owning window closes."""
        self._controller.unsubscribe_owner(self._id)

    # ── Cellier layer: model → widget ────────────────────────────────────────

    def _on_visual_changed(self, event) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        if event.field_name != "clim":
            return  # a different appearance field changed; nothing to do
        self._set_value(event.new_value)

    # ── Cellier layer: widget → model ────────────────────────────────────────

    def _on_slider_changed(self, value: tuple[float, float]) -> None:
        self._controller.incoming_events.emit(
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
