"""Show/hide toggle for any visual (Qt).

Layer 3: the whole class is a field name and a label.  Note that ``visible``
changes come back from the model on ``VisualVisibilityChangedEvent`` rather
than ``AppearanceChangedEvent``; that is handled once, in
``cellier.gui._appearance_fields``, so nothing here has to know.
"""

from __future__ import annotations

from typing import Any, ClassVar

from cellier.gui.qt.visuals._base import QtToggle


class QtVisibleToggle(QtToggle):
    """Show or hide the visual.

    ``visible`` exists on every appearance model, so this control applies to
    every visual type.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one group.
    initial_value :
        Starting value -- typically ``visual_model.appearance.visible``.
        Defaults to ``True``, matching the model default.
    parent :
        Optional Qt parent widget.
    """

    _field: ClassVar[str] = "visible"
    _label: ClassVar[str] = "Visible"
    _default_value: ClassVar[Any] = True
