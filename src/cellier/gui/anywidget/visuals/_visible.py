"""Show/hide toggle for any visual (anywidget).

Layer 3: the whole class is a field name and a label.  Note that ``visible``
changes come back from the model on ``VisualVisibilityChangedEvent`` rather
than ``AppearanceChangedEvent``; that is handled once, in
``cellier.gui._appearance_fields``, so nothing here has to know.
"""

from __future__ import annotations

from typing import Any, ClassVar

from cellier.gui.anywidget.visuals._base import AnywidgetToggle


class AnywidgetVisibleToggle(AnywidgetToggle):
    """Show or hide the visual.

    Mirrors ``QtVisibleToggle``.  ``visible`` exists on every appearance model,
    so this control applies to every visual type.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one group.
    initial_value :
        Starting value -- typically ``visual_model.appearance.visible``.
        Defaults to ``True``, matching the model default.
    """

    _field: ClassVar[str] = "visible"
    _label: ClassVar[str] = "Visible"
    _default_value: ClassVar[Any] = True
