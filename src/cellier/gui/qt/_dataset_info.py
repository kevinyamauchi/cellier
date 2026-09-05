"""Qt widget for displaying dataset metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cellier.gui._dataset_info import (
    DatasetInfo,
    MatrixSection,
    RowSection,
    dataset_info_from_path,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "DatasetInfo",
    "QtDatasetInfo",
    "QtOmeZarrMetadataWidget",
    "dataset_info_from_path",
]


class QtDatasetInfo:
    """Read-only display widget for dataset metadata.

    Shows ``(label, value)`` rows inside a ``superqt.QCollapsible``, the Qt
    idiom for the anywidget front end's ``<details>`` block.  Built from rows
    for the general case, or from a :class:`DatasetInfo` via :meth:`from_info`
    for the sectioned description a data store returns from its
    ``dataset_info()``, which adds nested blocks and matrix tables.

    Uses ``qtpy`` for PyQt6/PySide6 compatibility.  Follows the cellier v2
    widget pattern: a non-``QWidget`` class exposing a ``.widget`` property.
    It is a pure-output control with nothing on the bus, so it carries no
    ``changed``/``closed`` signals and is never passed to ``connect_widget``
    (``layout._shared.STATIC_CONTROL_KINDS``).

    Parameters
    ----------
    rows :
        ``(label, value)`` pairs to display, in order.  Both halves are
        coerced to ``str``.
    title :
        The name shown on the collapsible header.  Defaults to
        :data:`DEFAULT_TITLE`.
    parent :
        Optional Qt parent widget.
    """

    DEFAULT_TITLE = "Dataset info"
    """Name shown when no ``title=`` is given.

    The renderer passes the title from the shared control vocabulary; this is
    what a directly-constructed widget calls itself, and
    ``test_composite_default_titles_match_the_shared_vocabulary`` pins the two
    together.
    """

    def __init__(
        self,
        rows: Sequence[tuple[str, str]] = (),
        *,
        title: str | None = None,
        parent=None,
    ) -> None:
        from superqt import QCollapsible

        self._collapsible = QCollapsible(
            self.DEFAULT_TITLE if title is None else title, parent=parent
        )
        # collapsed by default

        # Sections are appended to the collapsible in the order the store
        # declared them, and nothing is hoisted.  There used to be a
        # standing top-level form here that unlabelled rows and *every*
        # matrix went into, while labelled row sections became nested
        # collapsibles appended after it -- so a matrix declared between two
        # labelled sections jumped above both, and the same DatasetInfo drew
        # in a different order than it did on the anywidget side
        # (``tests/gui/test_backend_parity.py``).
        self._section_labels: list[str] = []
        self._inline_form = None
        self._nested_by_label: dict[str, object] = {}

        if rows:
            self._add_rows(list(rows), label=None)

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self):
        """The ``QCollapsible`` widget to insert into a layout.

        Qt seam 1: replace with the backend element for other toolkits.
        """
        return self._collapsible

    def section_labels(self) -> list[str]:
        """The sections drawn, in the order they are drawn.

        The twin of reading ``AnywidgetDatasetInfo.sections``.  Both exist so
        the two front ends can be asserted to draw one ``DatasetInfo`` the
        same way, which is not otherwise checkable through a widget tree.

        An unlabelled section is named by its first row's label, matching
        what a reader actually sees at the top of the block.
        """
        return list(self._section_labels)

    # ── Section renderers ────────────────────────────────────────────────────

    def _form_widget(self, rows: Sequence[tuple[str, str]]):
        """A ``(label, value)`` block as one widget."""
        from qtpy.QtWidgets import QFormLayout, QLabel, QWidget

        content = QWidget()
        form = QFormLayout(content)
        form.setContentsMargins(4, 4, 4, 4)
        for label, value in rows:
            form.addRow(str(label), QLabel(str(value)))
        return content, form

    def _add_rows(self, rows: Sequence[tuple[str, str]], *, label: str | None) -> None:
        """Append a row block, nested in its own collapsible when labelled.

        Consecutive unlabelled blocks share one form so they read as a single
        list, which is what a store means by declaring rows with no label.
        A labelled block always starts a new one.
        """
        from qtpy.QtWidgets import QLabel

        if label is None:
            if self._inline_form is None:
                content, self._inline_form = self._form_widget(())
                self._collapsible.addWidget(content)
            for row_label, value in rows:
                self._inline_form.addRow(str(row_label), QLabel(str(value)))
            self._section_labels.append(str(rows[0][0]) if rows else "")
            return

        # A labelled section ends the run of inline rows: anything unlabelled
        # after it starts a fresh form below, rather than jumping back up.
        self._inline_form = None
        self._section_labels.append(label)
        self._add_nested(label, rows)

    def _add_nested(self, label: str, rows: Sequence[tuple[str, str]]) -> None:
        from superqt import QCollapsible

        nested = QCollapsible(label)
        content, _form = self._form_widget(rows)
        nested.addWidget(content)
        self._collapsible.addWidget(nested)
        self._nested_by_label[label] = nested

    def _add_row_section(self, section: RowSection) -> None:
        """Draw a :class:`RowSection`, nesting it when it carries a label."""
        self._add_rows(section.rows, label=section.label)
        if section.label is not None and not section.collapsed:
            self._nested_by_label[section.label].expand(animate=False)

    def _add_matrix_section(self, section: MatrixSection) -> None:
        """Draw a :class:`MatrixSection` as a non-editable table."""
        from qtpy.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem

        n_rows = len(section.row_labels)
        n_cols = len(section.col_labels)

        table = QTableWidget(n_rows, n_cols)
        table.setHorizontalHeaderLabels(section.col_labels)
        table.setVerticalHeaderLabels(section.row_labels)
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setFixedHeight(
            table.verticalHeader().length() + table.horizontalHeader().height() + 4
        )

        for row in range(n_rows):
            for col in range(n_cols):
                item = QTableWidgetItem(f"{section.matrix[row, col]:.4g}")
                table.setItem(row, col, item)

        # Appended where it was declared.  It used to go into the top-level
        # form, which put every matrix above every labelled row section
        # regardless of the order the store asked for.
        self._inline_form = None
        self._section_labels.append(section.label)
        _content, form = self._form_widget(())
        form.addRow(section.label, table)
        self._collapsible.addWidget(_content)

    @classmethod
    def from_info(
        cls,
        info: DatasetInfo,
        *,
        title: str | None = None,
        parent=None,
    ) -> QtDatasetInfo:
        """Build from a sectioned :class:`DatasetInfo`.

        Draws each section in the shape it declares: inline rows, a nested
        collapsible, or a matrix table.  Anything a store can say about
        itself is one of those three, so a new store type needs no change
        here.
        """
        self = cls(rows=(), title=title, parent=parent)

        for section in info.sections:
            if isinstance(section, RowSection):
                self._add_row_section(section)
            elif isinstance(section, MatrixSection):
                self._add_matrix_section(section)
            else:  # pragma: no cover - guards a future section type
                raise TypeError(f"Unknown dataset-info section: {section!r}")

        return self

    @classmethod
    def from_path(
        cls,
        zarr_path: str,
        *,
        multiscale_index: int = 0,
        series_index: int = 0,
        title: str | None = None,
        parent=None,
    ) -> QtDatasetInfo:
        """Construct directly from an OME-Zarr URI.

        Parameters
        ----------
        zarr_path :
            Root URI, e.g. ``"file:///data/image.ome.zarr"`` or
            ``"s3://bucket/image.ome.zarr"``.
        multiscale_index :
            Which ``multiscales[]`` entry to display. Defaults to 0.
        series_index :
            For Bf2Raw containers, which series to display. Defaults to 0.
        title :
            The name shown on the collapsible header.
        parent :
            Optional Qt parent widget.
        """
        info = dataset_info_from_path(
            zarr_path,
            multiscale_index=multiscale_index,
            series_index=series_index,
        )
        return cls.from_info(info, title=title, parent=parent)

    @classmethod
    def from_store(
        cls,
        store,
        *,
        title: str | None = None,
        parent=None,
    ) -> QtDatasetInfo:
        """Construct from any data store, by asking it to describe itself.

        The general entry point: every ``BaseDataStore`` implements
        ``dataset_info()``, so this works for points and graphs as much as
        for an OME-Zarr pyramid.
        """
        return cls.from_info(store.dataset_info(), title=title, parent=parent)


QtOmeZarrMetadataWidget = QtDatasetInfo
"""Deprecated alias.

The widget renders any ``(label, value)`` rows, not only OME-Zarr metadata,
so the name moved with the capability.
"""
