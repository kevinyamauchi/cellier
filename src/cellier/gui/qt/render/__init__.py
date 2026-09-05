"""Qt panels for the renderer's post-processing settings.

One panel per feature -- outlines, ambient occlusion, temporal
accumulation -- each a composite ``WidgetView`` driving one section of the
render config over the cellier bus.  Unlike the appearance widgets these
carry no entity id: render configuration belongs to the ``RenderManager``
rather than to any scene, visual or canvas.
"""

from cellier.gui.qt.render._ambient_occlusion import QtAmbientOcclusionControls
from cellier.gui.qt.render._base import QtRenderConfigPanel
from cellier.gui.qt.render._outline import QtOutlineControls
from cellier.gui.qt.render._per_visual import (
    QtLabelsOutlineControls,
    QtVisualOcclusionControls,
    QtVisualOutlineControls,
    QtVisualPickingControls,
    QtVisualRenderPanel,
)
from cellier.gui.qt.render._temporal import QtTemporalControls

__all__ = [
    "QtAmbientOcclusionControls",
    "QtLabelsOutlineControls",
    "QtOutlineControls",
    "QtRenderConfigPanel",
    "QtTemporalControls",
    "QtVisualOcclusionControls",
    "QtVisualOutlineControls",
    "QtVisualPickingControls",
    "QtVisualRenderPanel",
]
