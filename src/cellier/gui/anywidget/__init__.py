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
from cellier.gui.anywidget._toggle import AnywidgetDimToggle, make_dim_toggle_anywidget

__all__ = [
    "AnywidgetBox",
    "AnywidgetDatasetInfo",
    "AnywidgetDimToggle",
    "AnywidgetDimsPanel",
    "DatasetInfo",
    "dataset_info_from_path",
    "make_dim_toggle_anywidget",
]
