"""Notebook panels for the renderer's post-processing settings.

The anywidget twins of ``cellier.gui.qt.render``, drawing their controls
from the same shared spec so the two front ends stay in step.
"""

from cellier.gui.anywidget.render._base import AnywidgetRenderConfigPanel
from cellier.gui.anywidget.render._panels import (
    AnywidgetAmbientOcclusionControls,
    AnywidgetOutlineControls,
    AnywidgetTemporalControls,
)
from cellier.gui.anywidget.render._per_visual import (
    AnywidgetLabelsOutlineControls,
    AnywidgetVisualOcclusionControls,
    AnywidgetVisualOutlineControls,
    AnywidgetVisualPickingControls,
    AnywidgetVisualRenderPanel,
)

__all__ = [
    "AnywidgetAmbientOcclusionControls",
    "AnywidgetLabelsOutlineControls",
    "AnywidgetOutlineControls",
    "AnywidgetRenderConfigPanel",
    "AnywidgetTemporalControls",
    "AnywidgetVisualOcclusionControls",
    "AnywidgetVisualOutlineControls",
    "AnywidgetVisualPickingControls",
    "AnywidgetVisualRenderPanel",
]
