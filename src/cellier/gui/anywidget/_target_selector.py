"""The visual selector a controls dock shows when it can drive several visuals."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anywidget
import traitlets
from psygnal import Signal

from cellier.gui.anywidget._teardown import close_aux_widgets

if TYPE_CHECKING:
    from collections.abc import Sequence

# The generic select control every ``Literal`` appearance field already uses;
# it knows nothing about fields, only ``label`` / ``value`` / ``choices``.
_VISUALS_STATIC = Path(__file__).parent / "visuals" / "static"


class AnywidgetTargetSelector(anywidget.AnyWidget):
    """A labelled select choosing which configured visual a dock drives.

    Mirrors ``QtTargetSelector``.  Local UI state rather than a model control:
    it puts nothing on the bus and is never passed to ``connect_widget``.  The
    dock that owns it listens to ``selected``.

    Parameters
    ----------
    labels :
        The entries, in order.  They must be unique: the select reports the
        chosen entry by its text.
    index :
        The entry to show selected.
    title :
        The row's label.
    """

    _esm = _VISUALS_STATIC / "choice.js"
    _css = _VISUALS_STATIC / "choice.css"

    selected = Signal(int)

    label = traitlets.Unicode("").tag(sync=True)
    value = traitlets.Unicode("").tag(sync=True)
    choices = traitlets.List([]).tag(sync=True)

    def __init__(
        self, labels: Sequence[str], index: int = 0, *, title: str = "Visual"
    ) -> None:
        labels = list(labels)
        super().__init__(
            label=title, choices=labels, value=labels[index] if labels else ""
        )
        self._applying = False
        self.observe(self._on_value_change, names="value")

    @property
    def widget(self) -> AnywidgetTargetSelector:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    @property
    def labels(self) -> tuple[str, ...]:
        """The entries offered, in order."""
        return tuple(self.choices)

    @property
    def index(self) -> int:
        """The selected entry, or ``-1`` when there are none."""
        try:
            return list(self.choices).index(self.value)
        except ValueError:
            return -1

    def set_choices(self, labels: Sequence[str], index: int) -> None:
        """Replace the entries and the selection without emitting ``selected``."""
        labels = list(labels)
        self._applying = True
        try:
            with self.hold_sync():
                self.choices = labels
                self.value = labels[index] if labels else ""
        finally:
            self._applying = False

    def select(self, index: int) -> None:
        """Select entry *index* as a user would, emitting ``selected``."""
        self.value = self.choices[index]

    def close(self) -> None:
        """Release the widget."""
        close_aux_widgets(self)
        super().close()

    def _on_value_change(self, change) -> None:
        if self._applying:
            return
        try:
            index = list(self.choices).index(change["new"])
        except ValueError:
            return
        self.selected.emit(index)
