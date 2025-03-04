"""Example displaying and slicing a 4D volume in a Qt widget.

There is a 4D volume in a single array source.
"""

from qtpy import QtWidgets
from superqt import QLabeledSlider

from cellier.models.viewer import ViewerModel
from cellier.slicer.slicer import SlicerType
from cellier.viewer_controller import CellierController


class Main(QtWidgets.QWidget):
    """Example widget with viewer."""

    def __init__(self, viewer_config_path: str):
        super().__init__(None)
        self.resize(640, 480)

        # make the viewer model
        viewer_model = ViewerModel.from_json_file(viewer_config_path)

        # make the viewer
        self.viewer = CellierController(
            model=viewer_model, slicer_type=SlicerType.ASYNCHRONOUS, widget_parent=self
        )

        for cam in self.viewer._render_manager.cameras.values():
            cam.show_pos((20, 15, 15), up=(0, 1, 0))
            print(cam.get_state())

        # make the slider for the z axis in the 2D canvas
        self.t_slider_widget = QLabeledSlider(parent=self)
        self.t_slider_widget.valueChanged.connect(self._on_z_slider_changed)
        self.t_slider_widget.setMaximum(2)
        self.t_slider_widget.setValue(0)

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
            if coordinate_system.name == "scene_3d":
                dims_point = list(dims_manager.point)
                dims_point[0] = slider_value
                dims_manager.point = dims_point


CONFIG_PATH = "latent_volume_example_config.json"

app = QtWidgets.QApplication([])
m = Main(CONFIG_PATH)
m.show()


if __name__ == "__main__":
    app.exec()
