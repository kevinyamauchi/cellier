"""Qt visual-acceptance helpers (``plans/convenience_cleanup.md`` section 6.3).

``QWidget.grab()`` works under ``QT_QPA_PLATFORM=offscreen`` and returns real
pixels, so a rendered Qt dock can be checked by test rather than by eye.  These
helpers turn "run the dock once and look at it" into two assertions:

* :func:`control_labels` -- what the panel calls each control, **in order**, so
  a stage that adds, renames or reorders one shows up as a diff;
* :func:`assert_panel_renders` -- that the panel actually paints something, so
  a panel of correctly-titled but empty controls fails.

Import from a test module::

    from tests.convenience._qt_acceptance import assert_panel_renders, control_labels
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


def control_labels(container: QWidget) -> list[str]:
    """Return what *container* calls each control it holds, in layout order.

    The toolkit-neutral question "which controls does this panel contain, and
    what does it call them?" has two Qt answers, because a control is named in
    one of two ways (``plans/label_ownership_unification.md``):

    * a **single-control widget** carries its own label row, stamped with
      ``LABELLED_ROW_OBJECT_NAME``.  The row's first item is that label, and
      the walk does not descend past it -- a slider's own value readout and a
      colour picker's ``Alpha:`` are parts of the control, not names for it.
    * a **multi-row widget** (the AABB and volume-render groups) carries a
      ``QGroupBox``, whose title names the group.  The walk *does* descend
      into it, so a group holding labelled rows reports both.

    Anything else is scaffolding -- containers, stretches, the sub-labels
    inside a composite -- and contributes nothing.
    """
    from PySide6.QtWidgets import QGroupBox, QLabel

    from cellier.gui.qt.visuals._chrome import LABELLED_ROW_OBJECT_NAME

    labels: list[str] = []

    def _row_label(row: QWidget) -> str:
        layout = row.layout()
        if layout is None or layout.count() == 0:
            return ""
        first = layout.itemAt(0).widget()
        return first.text() if isinstance(first, QLabel) else ""

    def _walk(layout: QLayout | None) -> None:
        if layout is None:
            return
        for index in range(layout.count()):
            item = layout.itemAt(index)
            widget = item.widget()
            if widget is None:
                _walk(item.layout())
                continue
            if widget.objectName() == LABELLED_ROW_OBJECT_NAME:
                labels.append(_row_label(widget))
                continue
            if isinstance(widget, QGroupBox):
                labels.append(widget.title())
            _walk(widget.layout())

    _walk(container.layout())
    return labels


def control_labels_anywidget(widgets: list) -> list[str]:
    """Return what each built anywidget control calls itself, in order.

    The anywidget twin of :func:`control_labels`, and the reason the two front
    ends can be compared at all: both sides answer the same question about the
    same controls, so ``control_labels(qt_panel) == control_labels_anywidget(
    anywidget_widgets)`` is a real cross-toolkit assertion rather than two
    lists that happen to be written down side by side.

    A multi-row widget names itself with ``title``; a single-control widget
    with ``label``.  A widget carrying neither is one that cannot name itself,
    which is the state this plan removed -- it reports ``""`` rather than
    being skipped, so a regression shows up as a diff and not as a shorter
    list.
    """
    return [
        str(getattr(widget, "title", "") or getattr(widget, "label", ""))
        for widget in widgets
    ]


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
