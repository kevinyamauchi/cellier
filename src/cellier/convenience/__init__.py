"""Utilities to make it easier to construct Cellier viewers."""

from cellier.convenience._geometry import axis_ranges_from_viewer
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
from cellier.convenience._launch import launch, show
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
    "PointsMarkerAppearanceKwargs",
    "Viewer",
    "axis_ranges_from_viewer",
    "launch",
    "show",
]
