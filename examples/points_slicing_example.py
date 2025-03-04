"""Example displaying and slicing points in a Qt widget.

These are 4D points in a single array source rendered to two canvases:
a 3D view and a 2D view. The slider controls the position of the 2D slice.
"""

from qtpy import QtWidgets
from superqt import QLabeledSlider

from cellier.models.viewer import ViewerModel
from cellier.viewer_controller import CellierController


class Main(QtWidgets.QWidget):
    """Example widget with viewer."""

    def __init__(self, viewer_config_path: str):
        super().__init__(None)
        self.resize(640, 480)

        # make the viewer model
        viewer_model = ViewerModel.from_json_file(viewer_config_path)

        # make the viewer
        self.viewer = CellierController(model=viewer_model, widget_parent=self)

        for cam in self.viewer._render_manager.cameras.values():
            cam.show_pos((20, 15, 15), up=(0, 1, 0))
            print(cam.get_state())

        # make the slider for the z axis in the 2D canvas
        self.z_slider_widget = QLabeledSlider(parent=self)
        self.z_slider_widget.valueChanged.connect(self._on_z_slider_changed)
        self.z_slider_widget.setValue(25)

        layout = QtWidgets.QHBoxLayout()
        for canvas in self.viewer._canvas_widgets.values():
            # add the canvas widgets
            canvas.update()
            layout.addWidget(canvas)
        self.setLayout(layout)

    def _on_z_slider_changed(self, slider_value: int):
        for scene in self.viewer._model.scenes.scenes.values():
            dims_manager = scene.dims
            coordinate_system = dims_manager.coordinate_system
            if coordinate_system.name == "scene_2d":
                dims_point = list(dims_manager.point)
                dims_point[1] = slider_value
                dims_manager.point = dims_point


CONFIG_PATH = "points_example_config.json"

app = QtWidgets.QApplication([])
m = Main(CONFIG_PATH)
m.show()


if __name__ == "__main__":
    app.exec()
