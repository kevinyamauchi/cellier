"""Qt widget for displaying dataset metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cellier.gui._dataset_info import DatasetInfo, dataset_info_from_path

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
    for an OME-Zarr store, which adds the world-to-data affine matrix and the
    per-level shapes.

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
        from qtpy.QtWidgets import QFormLayout, QLabel, QWidget
        from superqt import QCollapsible

        self._collapsible = QCollapsible(
            self.DEFAULT_TITLE if title is None else title, parent=parent
        )
        # collapsed by default

        content = QWidget()
        self._form = QFormLayout(content)
        self._form.setContentsMargins(4, 4, 4, 4)
        self._collapsible.addWidget(content)

        for label, value in rows:
            self._form.addRow(str(label), QLabel(str(value)))

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self):
        """The ``QCollapsible`` widget to insert into a layout.

        Qt seam 1: replace with the backend element for other toolkits.
        """
        return self._collapsible

    @classmethod
    def from_info(
        cls,
        info: DatasetInfo,
        *,
        title: str | None = None,
        parent=None,
    ) -> QtDatasetInfo:
        """Build from a pre-extracted :class:`DatasetInfo`.

        Adds two blocks the plain row list has no shape for: the world-to-data
        affine as a table, and the per-level shapes in a nested collapsible.
        """
        from qtpy.QtWidgets import (
            QFormLayout,
            QHeaderView,
            QLabel,
            QTableWidget,
            QTableWidgetItem,
            QWidget,
        )
        from superqt import QCollapsible

        self = cls(
            rows=[
                ("File name", info.file_name),
                ("Type", info.zarr_type),
                ("Source", info.source),
            ],
            title=title,
            parent=parent,
        )

        # ── World-to-data transform matrix ──────────────────────────────────
        n = len(info.axis_names)
        headers = [*info.axis_names, "1"]  # homogeneous coordinate label
        row_labels = [*info.axis_names, ""]

        table = QTableWidget(n + 1, n + 1)
        table.setHorizontalHeaderLabels(headers)
        table.setVerticalHeaderLabels(row_labels)
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

        mat = info.world_to_data_matrix
        for row in range(n + 1):
            for col in range(n + 1):
                val = mat[row, col]
                text = f"{val:.4g}"
                item = QTableWidgetItem(text)
                table.setItem(row, col, item)

        self._form.addRow("World→data", table)

        # ── Shapes per scale level (nested collapsible) ──────────────────────
        shapes_collapsible = QCollapsible("scale shapes")
        # collapsed by default
        shapes_content = QWidget()
        shapes_form = QFormLayout(shapes_content)
        shapes_form.setContentsMargins(4, 4, 4, 4)
        shapes_collapsible.addWidget(shapes_content)

        axis_label = ", ".join(info.axis_names)
        for level_idx, shape in enumerate(info.scale_shapes):
            shape_str = " x ".join(str(s) for s in shape)
            shapes_form.addRow(f"level {level_idx} ({axis_label})", QLabel(shape_str))

        self._collapsible.addWidget(shapes_collapsible)
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


QtOmeZarrMetadataWidget = QtDatasetInfo
"""Deprecated alias.

The widget renders any ``(label, value)`` rows, not only OME-Zarr metadata,
so the name moved with the capability.
"""
