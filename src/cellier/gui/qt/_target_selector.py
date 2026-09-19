"""The visual selector a controls dock shows when it can drive several visuals."""

from __future__ import annotations

from typing import TYPE_CHECKING

from psygnal import Signal

if TYPE_CHECKING:
    from collections.abc import Sequence


class QtTargetSelector:
    """A labelled combo box choosing which configured visual a dock drives.

    Local UI state rather than a model control: it puts nothing on the bus and
    is never passed to ``connect_widget``.  The dock that owns it listens to
    ``selected``.

    Parameters
    ----------
    labels :
        The entries, in order.
    index :
        The entry to show selected.
    title :
        The row's label.
    parent :
        Optional Qt parent for the row.
    """

    selected = Signal(int)

    def __init__(
        self,
        labels: Sequence[str],
        index: int = 0,
        *,
        title: str = "Visual",
        parent=None,
    ) -> None:
        from qtpy.QtWidgets import QComboBox

        from cellier.gui.qt.visuals._chrome import labelled_row

        self._combo = QComboBox()
        self._combo.currentIndexChanged.connect(self._on_index_changed)
        self.set_choices(labels, index)
        self._widget = labelled_row(title, self._combo, parent)

    @property
    def widget(self):
        """The row to embed."""
        return self._widget

    @property
    def control(self):
        """The bare combo box."""
        return self._combo

    @property
    def labels(self) -> tuple[str, ...]:
        """The entries offered, in order."""
        return tuple(self._combo.itemText(i) for i in range(self._combo.count()))

    @property
    def index(self) -> int:
        """The selected entry, or ``-1`` when there are none."""
        return int(self._combo.currentIndex())

    def set_choices(self, labels: Sequence[str], index: int) -> None:
        """Replace the entries and the selection without emitting ``selected``."""
        self._combo.blockSignals(True)
        try:
            self._combo.clear()
            self._combo.addItems(list(labels))
            if labels:
                self._combo.setCurrentIndex(index)
        finally:
            self._combo.blockSignals(False)

    def select(self, index: int) -> None:
        """Select entry *index* as a user would, emitting ``selected``."""
        self._combo.setCurrentIndex(index)

    def close(self) -> None:
        """Release the row."""
        self._widget.setParent(None)
        self._widget.deleteLater()

    def _on_index_changed(self, index: int) -> None:
        if index >= 0:
            self.selected.emit(index)
