"""Utilities to make it easier to construct Cellier viewers."""

from cellier.convenience._geometry import (
    axis_ranges_from_ortho,
    axis_ranges_from_viewer,
)
from cellier.convenience._kwarg_dicts import (
    BaseAppearanceKwargs,
    BaseImageAppearanceKwargs,
    BaseLabelsAppearanceKwargs,
    ChannelAppearanceKwargs,
    InMemoryImageAppearanceKwargs,
    InMemoryLabelsAppearanceKwargs,
    LinesMemoryAppearanceKwargs,
    MeshFlatAppearanceKwargs,
    MeshPhongAppearanceKwargs,
    MultiscaleImageAppearanceKwargs,
    MultiscaleImageRenderConfigKwargs,
    MultiscaleLabelRenderConfigKwargs,
    MultiscaleLabelsAppearanceKwargs,
    PointsMarkerAppearanceKwargs,
)
from cellier.convenience._launch import display, launch, show
from cellier.convenience._ortho_viewer import OrthoViewer
from cellier.convenience._viewer import Viewer

__all__ = [
    "BaseAppearanceKwargs",
    "BaseImageAppearanceKwargs",
    "BaseLabelsAppearanceKwargs",
    "ChannelAppearanceKwargs",
    "InMemoryImageAppearanceKwargs",
    "InMemoryLabelsAppearanceKwargs",
    "LinesMemoryAppearanceKwargs",
    "MeshFlatAppearanceKwargs",
    "MeshPhongAppearanceKwargs",
    "MultiscaleImageAppearanceKwargs",
    "MultiscaleImageRenderConfigKwargs",
    "MultiscaleLabelRenderConfigKwargs",
    "MultiscaleLabelsAppearanceKwargs",
    "OrthoViewer",
    "PointsMarkerAppearanceKwargs",
    "Viewer",
    "axis_ranges_from_ortho",
    "axis_ranges_from_viewer",
    "display",
    "launch",
    "make_dim_toggle",
    "show",
]


def __getattr__(name: str):
    # Lazily expose make_dim_toggle so that importing cellier.convenience does
    # not require the optional anywidget dependency (it lives under
    # cellier.gui.anywidget, which imports anywidget at module load).
    if name == "make_dim_toggle":
        from cellier.gui.anywidget import make_dim_toggle

        return make_dim_toggle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
