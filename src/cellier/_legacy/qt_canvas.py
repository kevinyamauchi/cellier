"""Widget for the canvas."""

from qtpy.QtWidgets import QHBoxLayout, QtWidget
from wgpu.gui.qt import WgpuCanvas


class QtCanvas(QtWidget):
    """Qt widget for the canvas."""

    def __init__(self):
        self.wgpu_canvas = WgpuCanvas(parent=self)
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.layout().addWidget(self.wgpu_canvas)
