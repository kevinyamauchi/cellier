"""Colormap combobox wired to the cellier v2 event bus."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence
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


class QtColormapComboBox(VisualIdGroup):
    """Bidirectional colormap selector wired to the cellier v2 bus.

    Wraps a ``superqt.QColormapComboBox`` and keeps it in sync with
    ``MultiscaleImageAppearance.color_map`` via ``AppearanceChangedEvent``.  Follows the
    v2 widget pattern: one UUID per widget, source-ID echo filtering, and
    signal blocking when applying model-driven updates.

    Wire to the controller after construction::

        combo = QtColormapComboBox(visual_id, initial_colormap="grays")
        controller.connect_widget(combo, subscription_specs=combo.subscription_specs())

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``color_map`` field this widget controls.
        A sequence drives every listed visual in lock-step -- the
        ``OrthoViewer``'s four panel siblings (design section 8.1).
    initial_colormap :
        Starting colormap — typically ``visual_model.appearance.color_map``.
    parent :
        Optional Qt parent widget.
    """

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial_colormap,
        parent=None,
    ) -> None:
        from superqt import QColormapComboBox

        # ── Cellier layer ────────────────────────────────────────────────────
        self._id = uuid4()
        self._init_visual_ids(visual_id)

        # ── Qt seam 1: widget creation and signal wiring ─────────────────────
        self._combo = QColormapComboBox(parent)
        self._combo.setCurrentColormap(initial_colormap)
        self._combo.currentColormapChanged.connect(self._on_combo_changed)

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

    def add_colormaps(self, colormaps: Sequence[Any]) -> None:
        """Add colormaps to the combo box.

        Parameters
        ----------
        colormaps : Sequence[Any]
            Colormaps to add. Each item can be anything accepted by
            ``cmap.Colormap`` — e.g. a name string, a ``cmap.Colormap``
            instance, or a color-stop sequence.
        """
        self._combo.addColormaps(colormaps)

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Return the inbound subscription this widget requires.

        Pass the result to ``CellierController.connect_widget``.
        """
        return self._group_specs(AppearanceChangedEvent, self._on_visual_changed)

    # ── Cellier layer: model → widget ────────────────────────────────────────

    def _on_visual_changed(self, event) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        if event.field_name != "color_map":
            return  # a different appearance field changed; nothing to do
        self._set_value(event.new_value)

    # ── Cellier layer: widget → model ────────────────────────────────────────

    def _on_combo_changed(self, colormap) -> None:
        self._emit_group(AppearanceUpdateEvent, "color_map", colormap)

    # ── Qt seam 2: push value without re-firing currentColormapChanged ────────

    def _set_value(self, colormap) -> None:
        self._combo.blockSignals(True)
        self._combo.setCurrentColormap(colormap)
        self._combo.blockSignals(False)
