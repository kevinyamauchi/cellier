"""Example displaying and slicing points in a Qt widget."""

from qtpy import QtWidgets

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

        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        for canvas in self.viewer._canvas_widgets.values():
            # add the canvas widgets
            canvas.update()
            layout.addWidget(canvas)


CONFIG_PATH = "single_canvas_config.json"

app = QtWidgets.QApplication([])
m = Main(CONFIG_PATH)
m.show()


if __name__ == "__main__":
    app.exec()
