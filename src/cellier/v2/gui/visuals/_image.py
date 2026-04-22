"""Image-specific widgets wired to the cellier v2 event bus."""

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


class QtRenderModeComboBox:
    """Bidirectional render-mode selector wired to the cellier v2 bus.

    Wraps a ``QComboBox`` with ``"iso"`` and ``"mip"`` options and keeps it in
    sync with ``ImageAppearance.render_mode`` via ``AppearanceChangedEvent``.
    Follows the v2 widget pattern: one UUID per widget, source-ID echo
    filtering, and signal blocking when applying model-driven updates.

    Wire to the controller after construction::

        combo = QtRenderModeComboBox(visual_id, initial_render_mode="mip")
        controller.connect_widget(combo, subscription_specs=combo.subscription_specs())

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``render_mode`` field this widget controls.
    initial_render_mode :
        Starting value — typically ``visual_model.appearance.render_mode``.
    parent :
        Optional Qt parent widget.
    """

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID,
        *,
        initial_render_mode: str,
        parent=None,
    ) -> None:
        from qtpy.QtWidgets import QComboBox

        # ── Cellier layer ────────────────────────────────────────────────────
        self._id = uuid4()
        self._visual_id = visual_id

        # ── Qt seam 1: widget creation and signal wiring ─────────────────────
        self._combo = QComboBox(parent)
        self._combo.addItems(["iso", "mip"])
        self._combo.setCurrentText(initial_render_mode)
        self._combo.currentTextChanged.connect(self._on_combo_changed)

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self):
        """The Qt widget to insert into a layout.

        Qt seam 1: replace with the backend element for other toolkits.
        """
        return self._combo

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
        if event.field_name != "render_mode":
            return  # a different appearance field changed; nothing to do
        self._set_value(event.new_value)

    # ── Cellier layer: widget → model ────────────────────────────────────────

    def _on_combo_changed(self, text: str) -> None:
        self.changed.emit(
            AppearanceUpdateEvent(
                source_id=self._id,
                visual_id=self._visual_id,
                field="render_mode",
                value=text,
            )
        )

    # ── Qt seam 2: push value without re-firing currentTextChanged ───────────

    def _set_value(self, value: str) -> None:
        self._combo.blockSignals(True)
        self._combo.setCurrentText(value)
        self._combo.blockSignals(False)


class QtIsoThresholdSlider:
    """Bidirectional ISO threshold slider wired to the cellier v2 bus.

    Wraps a ``superqt.QLabeledDoubleSlider`` and keeps it in sync with
    ``ImageAppearance.iso_threshold`` via ``AppearanceChangedEvent``.
    Follows the v2 widget pattern: one UUID per widget, source-ID echo
    filtering, and signal blocking when applying model-driven updates.

    Wire to the controller after construction::

        slider = QtIsoThresholdSlider(visual_id, dtype_max=65535, initial_threshold=0.2)
        controller.connect_widget(slider, subscription_specs=slider.subscription_specs())

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``iso_threshold`` field this widget controls.
    dtype_max :
        Upper bound of the slider range — typically the maximum value of the
        volume's dtype (e.g. 65535 for uint16, 1.0 for float32).
    initial_threshold :
        Starting value — typically ``visual_model.appearance.iso_threshold``.
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
        dtype_max: float,
        initial_threshold: float,
        decimals: int = 2,
        parent=None,
    ) -> None:
        from qtpy.QtCore import Qt
        from superqt import QLabeledDoubleSlider

        # ── Cellier layer ────────────────────────────────────────────────────
        self._id = uuid4()
        self._visual_id = visual_id

        # ── Qt seam 1: widget creation and signal wiring ─────────────────────
        self._slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent)
        self._slider.setRange(0.0, dtype_max)
        self._slider.setValue(initial_threshold)
        self._slider.setDecimals(decimals)
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
        if event.field_name != "iso_threshold":
            return  # a different appearance field changed; nothing to do
        self._set_value(event.new_value)

    # ── Cellier layer: widget → model ────────────────────────────────────────

    def _on_slider_changed(self, value: float) -> None:
        self.changed.emit(
            AppearanceUpdateEvent(
                source_id=self._id,
                visual_id=self._visual_id,
                field="iso_threshold",
                value=value,
            )
        )

    # ── Qt seam 2: push value without re-firing valueChanged ─────────────────

    def _set_value(self, value: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(value)
        self._slider.blockSignals(False)


class QtVolumeRenderControls:
    """Combined render-mode and ISO-threshold widget wired to the cellier v2 bus.

    Contains a render-mode ``QComboBox`` (``"iso"`` / ``"mip"``) and a
    ``superqt.QLabeledDoubleSlider`` for the ISO threshold.  The threshold
    slider is only visible when the render mode is ``"iso"``.

    Both controls share one UUID so a single ``on_visual_changed`` subscription
    handles both fields.  Follows the v2 widget pattern: source-ID echo
    filtering and signal blocking on model-driven updates.

    Wire to the controller after construction::

        controls = QtVolumeRenderControls(visual_id, dtype_max=255, ...)
        controller.connect_widget(
            controls, subscription_specs=controls.subscription_specs()
        )

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``render_mode`` and ``iso_threshold`` fields
        this widget controls.
    dtype_max :
        Upper bound of the threshold slider — typically the dtype maximum
        (e.g. 65535 for uint16, 1.0 for float32).
    initial_render_mode :
        Starting render mode — typically ``visual_model.appearance.render_mode``.
    initial_threshold :
        Starting threshold — typically ``visual_model.appearance.iso_threshold``.
    decimals :
        Number of decimal places shown in the threshold slider label.  Use
        ``0`` for integer dtypes and ``2`` (or similar) for float data.
        Default is ``2``.
    parent :
        Optional Qt parent widget.
    """

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID,
        *,
        dtype_max: float,
        initial_render_mode: str,
        initial_threshold: float,
        decimals: int = 2,
        parent=None,
    ) -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QComboBox, QFormLayout, QWidget
        from superqt import QLabeledDoubleSlider

        # ── Cellier layer ────────────────────────────────────────────────────
        self._id = uuid4()
        self._visual_id = visual_id

        # ── Qt seam 1: widget creation and signal wiring ─────────────────────
        self._container = QWidget(parent)
        layout = QFormLayout(self._container)
        layout.setContentsMargins(0, 0, 0, 0)

        self._combo = QComboBox()
        self._combo.addItems(["iso", "mip"])
        self._combo.setCurrentText(initial_render_mode)
        self._combo.currentTextChanged.connect(self._on_combo_changed)
        layout.addRow("Mode", self._combo)

        self._slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0.0, dtype_max)
        self._slider.setValue(initial_threshold)
        self._slider.setDecimals(decimals)
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addRow("Threshold", self._slider)

        # Show threshold row only in ISO mode.
        self._slider.setVisible(initial_render_mode == "iso")
        layout.labelForField(self._slider).setVisible(initial_render_mode == "iso")

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
                event_type=AppearanceChangedEvent,
                handler=self._on_visual_changed,
                entity_id=self._visual_id,
            )
        ]

    # ── Cellier layer: model → widget ────────────────────────────────────────

    def _on_visual_changed(self, event) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        if event.field_name == "render_mode":
            self._set_render_mode(event.new_value)
        elif event.field_name == "iso_threshold":
            self._set_threshold(event.new_value)

    # ── Cellier layer: widget → model ────────────────────────────────────────

    def _on_combo_changed(self, text: str) -> None:
        self._update_threshold_visibility(text)
        self.changed.emit(
            AppearanceUpdateEvent(
                source_id=self._id,
                visual_id=self._visual_id,
                field="render_mode",
                value=text,
            )
        )

    def _on_slider_changed(self, value: float) -> None:
        self.changed.emit(
            AppearanceUpdateEvent(
                source_id=self._id,
                visual_id=self._visual_id,
                field="iso_threshold",
                value=value,
            )
        )

    # ── Qt seam 2: push values without re-firing signals ─────────────────────

    def _set_render_mode(self, value: str) -> None:
        self._combo.blockSignals(True)
        self._combo.setCurrentText(value)
        self._combo.blockSignals(False)
        self._update_threshold_visibility(value)

    def _set_threshold(self, value: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(value)
        self._slider.blockSignals(False)

    def _update_threshold_visibility(self, render_mode: str) -> None:
        visible = render_mode == "iso"
        self._slider.setVisible(visible)
        layout = self._container.layout()
        layout.labelForField(self._slider).setVisible(visible)
