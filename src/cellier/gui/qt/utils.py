"""Utility functions for generating Qt widgets."""

from typing import Dict, Optional

from qtpy.QtWidgets import QWidget

# from rendercanvas.qt import QRenderWidget
from wgpu.gui.qt import WgpuCanvas

from cellier.models.viewer import ViewerModel


def construct_qt_canvases_from_model(
    viewer_model: ViewerModel, parent: Optional[QWidget] = None
) -> Dict[str, WgpuCanvas]:
    """Construct Qt PyGFX canvas widgets from the viewer model.

    Parameters
    ----------
    viewer_model : ViewerModel
        The viewer model to construct the Qt canvases from.

    parent : Optional[QWidget]
        The parent for the constructed Qt canvases.
        Default value is None.

    Returns
    -------
    Dict[str, WgpuCanvas]
        A dictionary where the canvas IDs are the keys and
        The constructed Qt canvases are the values.
    """
    canvas_widgets = {}
    for scene_model in viewer_model.scenes.scenes.values():
        for canvas_model in scene_model.canvases.values():
            # todo switch to rendercanvas
            # canvas_widgets[canvas_model.id] = QRenderWidget(parent=parent)
            canvas_widgets[canvas_model.id] = WgpuCanvas(parent=parent)

    return canvas_widgets
