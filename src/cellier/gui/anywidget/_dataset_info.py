"""anywidget widget for displaying dataset metadata.

Mirrors ``QtDatasetInfo``: a read-only display, not part of the ``WidgetView``
bus contract (no ``changed``/``closed``/``subscription_specs``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anywidget
import traitlets

from cellier.gui._dataset_info import DatasetInfo, dataset_info_from_path

if TYPE_CHECKING:
    from collections.abc import Sequence

_STATIC = Path(__file__).parent / "static"

__all__ = ["AnywidgetDatasetInfo", "DatasetInfo", "dataset_info_from_path"]


class AnywidgetDatasetInfo(anywidget.AnyWidget):
    """Read-only display widget for dataset metadata.

    Shows ``(label, value)`` rows as a table inside a collapsible
    ``<details>`` block.

    The rows cross to the front end as data and are written with
    ``textContent``, not as markup: a value carrying ``<`` or ``&`` -- a path,
    a dtype, anything read off a store -- is displayed, never parsed.

    Parameters
    ----------
    rows :
        ``(label, value)`` pairs to display, in order.  Both halves are
        coerced to ``str``.  An empty sequence hides the block.
    """

    _esm = _STATIC / "dataset_info.js"
    _css = _STATIC / "dataset_info.css"

    DEFAULT_TITLE = "Dataset info"
    """Name shown when no ``title=`` is given.

    The renderer passes the title from the shared control vocabulary; this is
    what a directly-constructed widget calls itself, and
    ``test_composite_default_titles_match_the_shared_vocabulary`` pins the two
    together.
    """

    title = traitlets.Unicode(DEFAULT_TITLE).tag(sync=True)
    """What this control calls itself, drawn by its own front end.

    A control names itself rather than being named by whatever lays it out
    (``plans/label_ownership_unification.md``), which is what lets the dock
    stack controls and stop.
    """

    rows = traitlets.List().tag(sync=True)
    """``[[label, value], ...]``, the synced form of the ``rows`` argument."""

    def __init__(self, rows: Sequence[tuple[str, str]] = (), **kwargs) -> None:
        super().__init__(
            rows=[[str(label), str(value)] for label, value in rows], **kwargs
        )

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self) -> AnywidgetDatasetInfo:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self
