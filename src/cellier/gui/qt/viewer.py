"""Qt widget for the viewer."""

from PyQt5.QtWidgets import QLabel, QVBoxLayout
from qtpy.QtWidgets import QWidget
from wgpu.gui.qt import WgpuCanvas

from cellier.models.viewer import ViewerModel


class QtViewer(QWidget):
    """Qt widget for the viewer.

    This contains all the canvases and the dim sliders.
    """

    def __init__(self, viewer_model: ViewerModel) -> None:
        super().__init__()

        # make the canvas widgets
        canvas_widgets = {}
        for scene_model in viewer_model.scenes.scenes:
            for canvas_model in scene_model.canvases:
                canvas_widgets[canvas_model.id] = WgpuCanvas()
        self._canvases = canvas_widgets

        # make the layout
        self.setLayout(QVBoxLayout())
        for canvas_widget in self._canvases.values():
            # add the canvas widgets
            self.layout().addWidget(QLabel(canvas_widget))
