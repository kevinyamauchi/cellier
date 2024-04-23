"""Qt widget for the viewer."""

from typing import Dict

from qtpy.QtWidgets import QVBoxLayout, QWidget
from wgpu.gui.qt import WgpuCanvas

from cellier.models.viewer import ViewerModel


class QtViewer(QWidget):
    """Qt widget for the viewer.

    This contains all the canvases and the dim sliders.
    """

    def __init__(self, viewer_model: ViewerModel, parent=None) -> None:
        super().__init__(parent=parent)

        # make the canvas widgets
        canvas_widgets = {}
        for scene_model in viewer_model.scenes.scenes:
            for canvas_model in scene_model.canvases:
                canvas_widgets[canvas_model.id] = WgpuCanvas(parent=self)
        self._canvases = canvas_widgets

        # make the layout
        self.setLayout(QVBoxLayout())
        for canvas_widget in self._canvases.values():
            # add the canvas widgets
            self.layout().addWidget(canvas_widget)

    @property
    def canvases(self) -> Dict[str, WgpuCanvas]:
        """The canvas widgets.

        Returns
        -------
        Dict[str, WgpuCanvas]
            Dictionary where the keys are the canvas IDs and
            the values are the canvas widgets.
        """
        return self._canvases
