"""Qt visual-acceptance helpers (``plans/convenience_cleanup.md`` section 6.3).

``QWidget.grab()`` works under ``QT_QPA_PLATFORM=offscreen`` and returns real
pixels, so a rendered Qt dock can be checked by test rather than by eye.  These
helpers turn "run the dock once and look at it" into two assertions:

* :func:`group_titles` -- which controls the panel contains, **in order**, so a
  stage that adds or reorders a group shows up as a diff;
* :func:`assert_panel_renders` -- that the panel actually paints something, so
  a panel of correctly-titled but empty groups fails.

Import from a test module::

    from tests.convenience._qt_acceptance import assert_panel_renders, group_titles
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QLayout, QWidget

# Sampling stride for the rendered pixmap.  The panel is a few hundred pixels
# on a side, so every third pixel is plenty to tell "painted" from "blank"
# while keeping the scan cheap.
_SAMPLE_STRIDE = 3

# A blank panel is one flat background colour (plus, at most, antialiasing on
# a border).  Anything that drew a label, a border and a control clears this
# comfortably -- a fully populated appearance dock scores ~190.
_MIN_DISTINCT_COLORS = 8


def group_titles(container: QWidget) -> list[str]:
    """Return the ``QGroupBox`` titles inside *container*, in layout order.

    Walks the layout tree rather than using ``findChildren``, so the result is
    the order the user sees top-to-bottom, not the order of construction.
    Nested groups follow their parent.
    """
    from PySide6.QtWidgets import QGroupBox

    titles: list[str] = []

    def _walk(layout: QLayout | None) -> None:
        if layout is None:
            return
        for index in range(layout.count()):
            item = layout.itemAt(index)
            widget = item.widget()
            if widget is None:
                _walk(item.layout())
                continue
            if isinstance(widget, QGroupBox):
                titles.append(widget.title())
            _walk(widget.layout())

    _walk(container.layout())
    return titles


def panel_distinct_colors(container: QWidget) -> int:
    """Grab *container* offscreen and count the distinct sampled pixel colours.

    ``grab()`` renders through the real paint pipeline even on the offscreen
    platform plugin, so this reports what the panel would actually look like.
    """
    container.resize(container.sizeHint())
    pixmap = container.grab()
    assert not pixmap.isNull(), "QWidget.grab() returned a null pixmap"

    image = pixmap.toImage()
    colors = {
        image.pixel(x, y)
        for y in range(0, image.height(), _SAMPLE_STRIDE)
        for x in range(0, image.width(), _SAMPLE_STRIDE)
    }
    return len(colors)


def assert_panel_renders(container: QWidget | None) -> None:
    """Assert *container* exists, has a non-zero size, and paints content."""
    assert container is not None, "expected a rendered panel, got None"

    size = container.sizeHint()
    assert size.width() > 0 and size.height() > 0, f"panel has empty size {size}"

    n_colors = panel_distinct_colors(container)
    assert n_colors >= _MIN_DISTINCT_COLORS, (
        f"panel painted only {n_colors} distinct colours "
        f"(< {_MIN_DISTINCT_COLORS}); it looks blank"
    )
