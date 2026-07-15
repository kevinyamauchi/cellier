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
    ChannelControlsKwargs,
    InMemoryImageAppearanceKwargs,
    InMemoryImageControlsKwargs,
    InMemoryLabelsAppearanceKwargs,
    LinesMemoryAppearanceKwargs,
    MeshFlatAppearanceKwargs,
    MeshPhongAppearanceKwargs,
    MultiscaleImageAppearanceKwargs,
    MultiscaleImageControlsKwargs,
    MultiscaleImageRenderConfigKwargs,
    MultiscaleLabelRenderConfigKwargs,
    MultiscaleLabelsAppearanceKwargs,
    PointsMarkerAppearanceKwargs,
    SidecarKwargs,
)
from cellier.convenience._launch import DisplayHandle, display, launch, run, show
from cellier.convenience._ortho_viewer import OrthoViewer
from cellier.convenience._sidecar import SidecarOptions
from cellier.convenience._viewer import Viewer
from cellier.convenience.layout import (
    AppearanceControls,
    ChannelControls,
    Grid,
    HStack,
    Layout,
    VStack,
)

__all__ = [
    "AppearanceControls",
    "BaseAppearanceKwargs",
    "BaseImageAppearanceKwargs",
    "BaseLabelsAppearanceKwargs",
    "ChannelAppearanceKwargs",
    "ChannelControls",
    "ChannelControlsKwargs",
    "DisplayHandle",
    "Grid",
    "HStack",
    "InMemoryImageAppearanceKwargs",
    "InMemoryImageControlsKwargs",
    "InMemoryLabelsAppearanceKwargs",
    "Layout",
    "LinesMemoryAppearanceKwargs",
    "MeshFlatAppearanceKwargs",
    "MeshPhongAppearanceKwargs",
    "MultiscaleImageAppearanceKwargs",
    "MultiscaleImageControlsKwargs",
    "MultiscaleImageRenderConfigKwargs",
    "MultiscaleLabelRenderConfigKwargs",
    "MultiscaleLabelsAppearanceKwargs",
    "OrthoViewer",
    "PointsMarkerAppearanceKwargs",
    "SidecarKwargs",
    "SidecarOptions",
    "VStack",
    "Viewer",
    "axis_ranges_from_ortho",
    "axis_ranges_from_viewer",
    "display",
    "launch",
    "run",
    "show",
]
