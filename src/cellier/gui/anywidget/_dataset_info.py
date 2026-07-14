"""anywidget widget for displaying OME-Zarr dataset metadata.

Mirrors ``QtOmeZarrMetadataWidget``: a read-only display, not part of the
``WidgetView`` bus contract (no ``changed``/``closed``/``subscription_specs``).
"""

from __future__ import annotations

from pathlib import Path

import anywidget
import traitlets

from cellier.gui._dataset_info import DatasetInfo, dataset_info_from_path

_STATIC = Path(__file__).parent / "static"

__all__ = ["AnywidgetDatasetInfo", "DatasetInfo", "dataset_info_from_path"]


def _format_dataset_info_html(info: DatasetInfo) -> str:
    """Render a :class:`DatasetInfo` as the HTML block the JS injects verbatim."""
    n = len(info.axis_names)
    headers = [*info.axis_names, "1"]
    row_labels = [*info.axis_names, ""]
    mat = info.world_to_data_matrix

    matrix_rows = []
    for row in range(n + 1):
        cells = "".join(f"<td>{mat[row, col]:.4g}</td>" for col in range(n + 1))
        matrix_rows.append(f"<tr><th>{row_labels[row]}</th>{cells}</tr>")
    matrix_html = (
        "<table><tr><th></th>"
        + "".join(f"<th>{h}</th>" for h in headers)
        + "</tr>"
        + "".join(matrix_rows)
        + "</table>"
    )

    axis_label = ", ".join(info.axis_names)
    shape_rows = "".join(
        f"<tr><td>level {i} ({axis_label})</td><td>{' x '.join(str(s) for s in shape)}</td></tr>"
        for i, shape in enumerate(info.scale_shapes)
    )
    shapes_html = f"<table>{shape_rows}</table>"

    return (
        "<div class='cellier-dataset-info-rows'>"
        f"<div><b>File name</b>: {info.file_name}</div>"
        f"<div><b>Type</b>: {info.zarr_type}</div>"
        f"<div><b>Source</b>: {info.source}</div>"
        "<div><b>World&rarr;data</b></div>"
        f"{matrix_html}"
        "<div><b>Scale shapes</b></div>"
        f"{shapes_html}"
        "</div>"
    )


class AnywidgetDatasetInfo(anywidget.AnyWidget):
    """Read-only display widget for OME-Zarr dataset metadata.

    Shows file name, type, storage source, world-to-data affine matrix, and
    the shape of each resolution level inside a collapsible ``<details>``
    block.

    Parameters
    ----------
    dataset_info :
        Pre-formatted HTML for the detail block.  Empty string hides it.
        Use :meth:`from_info` or :meth:`from_path` to build this from a
        :class:`DatasetInfo` instead of formatting it by hand.
    """

    _esm = _STATIC / "dataset_info.js"
    _css = _STATIC / "dataset_info.css"

    dataset_info = traitlets.Unicode("").tag(sync=True)

    def __init__(self, dataset_info: str = "", **kwargs) -> None:
        super().__init__(dataset_info=str(dataset_info), **kwargs)

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self) -> AnywidgetDatasetInfo:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    @classmethod
    def from_info(cls, info: DatasetInfo) -> AnywidgetDatasetInfo:
        """Build directly from a pre-extracted :class:`DatasetInfo`."""
        return cls(dataset_info=_format_dataset_info_html(info))

    @classmethod
    def from_path(
        cls,
        zarr_path: str,
        *,
        multiscale_index: int = 0,
        series_index: int = 0,
    ) -> AnywidgetDatasetInfo:
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
        """
        info = dataset_info_from_path(
            zarr_path,
            multiscale_index=multiscale_index,
            series_index=series_index,
        )
        return cls.from_info(info)
