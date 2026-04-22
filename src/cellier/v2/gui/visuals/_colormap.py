"""Colormap combobox wired to the cellier v2 event bus."""

from __future__ import annotations

from uuid import uuid4

from cellier.v2.events import AppearanceUpdateEvent


class QtColormapComboBox:
    """Bidirectional colormap selector wired to the cellier v2 bus.

    Wraps a ``superqt.QColormapComboBox`` and keeps it in sync with
    ``ImageAppearance.color_map`` via ``AppearanceChangedEvent``.  Follows the
    v2 widget pattern: one UUID per widget, source-ID echo filtering, and
    signal blocking when applying model-driven updates.

    Parameters
    ----------
    controller :
        The ``CellierController`` instance.
    visual_id :
        UUID of the visual whose ``color_map`` field this widget controls.
    initial_colormap :
        Starting colormap — typically ``visual_model.appearance.color_map``.
    parent :
        Optional Qt parent widget.
    """

    def __init__(
        self,
        controller,
        visual_id,
        *,
        initial_colormap,
        parent=None,
    ) -> None:
        from superqt import QColormapComboBox

        # ── Cellier layer ────────────────────────────────────────────────────
        self._id = uuid4()
        self._controller = controller
        self._visual_id = visual_id

        # ── Qt seam 1: widget creation and signal wiring ─────────────────────
        self._combo = QColormapComboBox(parent)
        self._combo.setCurrentColormap(initial_colormap)
        self._combo.currentColormapChanged.connect(self._on_combo_changed)

        controller.on_visual_changed(
            visual_id, self._on_visual_changed, owner_id=self._id
        )

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self):
        """The Qt widget to insert into a layout.

        Qt seam 1: replace with the backend element for other toolkits.
        """
        return self._combo

    def close(self) -> None:
        """Unsubscribe from the bus.  Call when the owning window closes."""
        self._controller.unsubscribe_owner(self._id)

    # ── Cellier layer: model → widget ────────────────────────────────────────

    def _on_visual_changed(self, event) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        if event.field_name != "color_map":
            return  # a different appearance field changed; nothing to do
        self._set_value(event.new_value)

    # ── Cellier layer: widget → model ────────────────────────────────────────

    def _on_combo_changed(self, colormap) -> None:
        self._controller.incoming_events.emit(
            AppearanceUpdateEvent(
                source_id=self._id,
                visual_id=self._visual_id,
                field="color_map",
                value=colormap,
            )
        )

    # ── Qt seam 2: push value without re-firing currentColormapChanged ────────

    def _set_value(self, colormap) -> None:
        self._combo.blockSignals(True)
        self._combo.setCurrentColormap(colormap)
        self._combo.blockSignals(False)
