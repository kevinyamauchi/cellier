"""cellier anywidget (notebook) GUI widgets.

Importing this package requires the optional ``anywidget`` dependency
(``pip install 'cellier[anywidget]'``).
"""

from cellier.gui.anywidget._container import AnywidgetBox
from cellier.gui.anywidget._dataset_info import (
    AnywidgetDatasetInfo,
    DatasetInfo,
    dataset_info_from_path,
)
from cellier.gui.anywidget._dims_panel import AnywidgetDimsPanel

__all__ = [
    "AnywidgetBox",
    "AnywidgetDatasetInfo",
    "AnywidgetDimsPanel",
    "DatasetInfo",
    "dataset_info_from_path",
]
