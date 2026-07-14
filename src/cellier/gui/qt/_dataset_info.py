"""Qt widget for displaying OME-Zarr dataset metadata."""

from __future__ import annotations

from cellier.gui._dataset_info import DatasetInfo, dataset_info_from_path

__all__ = ["DatasetInfo", "QtOmeZarrMetadataWidget", "dataset_info_from_path"]


class QtOmeZarrMetadataWidget:
    """Read-only display widget for OME-Zarr dataset metadata.

    Shows file name, type, storage source, world-to-data affine matrix, and
    the shape of each resolution level inside a ``superqt.QCollapsible``
    titled ``"dataset"``.

    Uses ``qtpy`` for PyQt6/PySide6 compatibility.  Follows the cellier v2
    widget pattern: a non-``QWidget`` class exposing a ``.widget`` property.

    Parameters
    ----------
    info :
        Pre-extracted :class:`DatasetInfo`.  Use
        :func:`dataset_info_from_path` to build one from a zarr URI.
    parent :
        Optional Qt parent widget.
    """

    def __init__(self, info: DatasetInfo, *, parent=None) -> None:
        from qtpy.QtWidgets import (
            QFormLayout,
            QHeaderView,
            QLabel,
            QTableWidget,
            QTableWidgetItem,
            QWidget,
        )
        from superqt import QCollapsible

        self._collapsible = QCollapsible("dataset info", parent=parent)
        # collapsed by default

        content = QWidget()
        form = QFormLayout(content)
        form.setContentsMargins(4, 4, 4, 4)
        self._collapsible.addWidget(content)

        # ── File name ────────────────────────────────────────────────────────
        form.addRow("File name", QLabel(info.file_name))

        # ── Type ────────────────────────────────────────────────────────────
        form.addRow("Type", QLabel(info.zarr_type))

        # ── Source ──────────────────────────────────────────────────────────
        form.addRow("Source", QLabel(info.source))

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

        form.addRow("World→data", table)

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

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self):
        """The ``QCollapsible`` widget to insert into a layout.

        Qt seam 1: replace with the backend element for other toolkits.
        """
        return self._collapsible

    @classmethod
    def from_path(
        cls,
        zarr_path: str,
        *,
        multiscale_index: int = 0,
        series_index: int = 0,
        parent=None,
    ) -> QtOmeZarrMetadataWidget:
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
        parent :
            Optional Qt parent widget.
        """
        info = dataset_info_from_path(
            zarr_path,
            multiscale_index=multiscale_index,
            series_index=series_index,
        )
        return cls(info, parent=parent)
