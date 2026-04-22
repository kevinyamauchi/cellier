"""AABB (axis-aligned bounding box) control widget wired to the cellier v2 event bus."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from psygnal import Signal

from cellier.v2.events import AABBChangedEvent, AABBUpdateEvent, SubscriptionSpec

if TYPE_CHECKING:
    from uuid import UUID


class QtAABBWidget:
    """Bidirectional AABB parameter controls wired to the cellier v2 bus.

    Wraps three Qt sub-controls — a ``QCheckBox`` for *enabled*, a
    ``QDoubleSpinBox`` for *line_width*, and a color swatch + button for
    *color* — and keeps them in sync with ``AABBParams`` via
    ``AABBChangedEvent``.  Follows the v2 widget pattern: one UUID per
    widget, source-ID echo filtering, and signal blocking when applying
    model-driven updates.

    Wire to the controller after construction::

        aabb = QtAABBWidget(visual_id, initial_enabled=True, ...)
        controller.connect_widget(aabb, subscription_specs=aabb.subscription_specs())

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``aabb`` params this widget controls.
    initial_enabled :
        Starting value for the *enabled* checkbox.
    initial_line_width :
        Starting value for the *line_width* spinbox.
    initial_color :
        Starting CSS color string for the color swatch (e.g. ``"#ff00ff"``).
    parent :
        Optional Qt parent widget.
    """

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID,
        *,
        initial_enabled: bool = False,
        initial_line_width: float = 2.0,
        initial_color: str = "#ffffff",
        parent=None,
    ) -> None:
        from qtpy.QtWidgets import (
            QCheckBox,
            QDoubleSpinBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        # ── Cellier layer ────────────────────────────────────────────────────
        self._id = uuid4()
        self._visual_id = visual_id
        self._current_color = initial_color

        # ── Qt seam 1: container ─────────────────────────────────────────────
        self._container = QWidget(parent)
        layout = QVBoxLayout(self._container)
        layout.setContentsMargins(0, 0, 0, 0)

        # enabled checkbox
        self._enabled_check = QCheckBox("Show bounding box")
        self._enabled_check.setChecked(initial_enabled)
        self._enabled_check.toggled.connect(self._on_enabled_changed)
        layout.addWidget(self._enabled_check)

        # line width row
        lw_row = QWidget()
        lw_layout = QHBoxLayout(lw_row)
        lw_layout.setContentsMargins(0, 0, 0, 0)
        lw_layout.addWidget(QLabel("Line width (px):"))
        self._line_width_spin = QDoubleSpinBox()
        self._line_width_spin.setRange(0.5, 20.0)
        self._line_width_spin.setSingleStep(0.5)
        self._line_width_spin.setDecimals(1)
        self._line_width_spin.setValue(initial_line_width)
        self._line_width_spin.valueChanged.connect(self._on_line_width_changed)
        lw_layout.addWidget(self._line_width_spin)
        layout.addWidget(lw_row)

        # color row: swatch + button
        color_row = QWidget()
        color_layout = QHBoxLayout(color_row)
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_layout.addWidget(QLabel("Color:"))
        self._color_swatch = QWidget()
        self._color_swatch.setFixedSize(24, 24)
        self._color_swatch.setToolTip("Current AABB color")
        color_layout.addWidget(self._color_swatch)
        self._color_btn = QPushButton("Choose...")
        self._color_btn.clicked.connect(self._on_color_btn_clicked)
        color_layout.addWidget(self._color_btn)
        color_layout.addStretch()
        layout.addWidget(color_row)

        # apply initial swatch color (no signal to block)
        self._apply_swatch(initial_color)

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self):
        """The Qt widget to insert into a layout.

        Qt seam 1: replace with the backend element for other toolkits.
        """
        return self._container

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Return the inbound subscription this widget requires.

        Pass the result to ``CellierController.connect_widget``.
        """
        return [
            SubscriptionSpec(
                event_type=AABBChangedEvent,
                handler=self._on_aabb_changed,
                entity_id=self._visual_id,
            )
        ]

    # ── Cellier layer: model → widget ────────────────────────────────────────

    def _on_aabb_changed(self, event) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        if event.field_name == "enabled":
            self._set_enabled(event.new_value)
        elif event.field_name == "line_width":
            self._set_line_width(event.new_value)
        elif event.field_name == "color":
            self._set_color(event.new_value)

    # ── Cellier layer: widget → model ────────────────────────────────────────

    def _on_enabled_changed(self, value: bool) -> None:
        self.changed.emit(
            AABBUpdateEvent(
                source_id=self._id,
                visual_id=self._visual_id,
                field="enabled",
                value=value,
            )
        )

    def _on_line_width_changed(self, value: float) -> None:
        self.changed.emit(
            AABBUpdateEvent(
                source_id=self._id,
                visual_id=self._visual_id,
                field="line_width",
                value=value,
            )
        )

    def _on_color_btn_clicked(self) -> None:
        from qtpy.QtGui import QColor
        from qtpy.QtWidgets import QColorDialog

        initial = QColor(self._current_color)
        color = QColorDialog.getColor(initial, self._container, "Choose AABB color")
        if not color.isValid():
            return  # user cancelled
        css = color.name()  # e.g. "#ff00ff"
        self.changed.emit(
            AABBUpdateEvent(
                source_id=self._id,
                visual_id=self._visual_id,
                field="color",
                value=css,
            )
        )
        # Update local cache and swatch immediately — echo filtering will
        # suppress the round-tripped AABBChangedEvent.
        self._current_color = css
        self._apply_swatch(css)

    # ── Qt seam 2: push values without re-firing change signals ─────────────

    def _set_enabled(self, value: bool) -> None:
        self._enabled_check.blockSignals(True)
        self._enabled_check.setChecked(value)
        self._enabled_check.blockSignals(False)

    def _set_line_width(self, value: float) -> None:
        self._line_width_spin.blockSignals(True)
        self._line_width_spin.setValue(value)
        self._line_width_spin.blockSignals(False)

    def _set_color(self, value: str) -> None:
        # Swatch has no change signal — no blocking needed.
        self._current_color = value
        self._apply_swatch(value)

    def _apply_swatch(self, css_color: str) -> None:
        """Update the color swatch background without touching any signals."""
        self._color_swatch.setStyleSheet(
            f"background-color: {css_color}; border: 1px solid #888;"
        )
