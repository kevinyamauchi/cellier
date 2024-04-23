"""Example displaying viewer in a Qt widget."""

import random

import numpy as np
import pygfx as gfx
from qtpy import QtWidgets
from wgpu.gui.qt import WgpuCanvas

from cellier.models.data_stores.mesh import MeshMemoryStore
from cellier.models.data_streams.mesh import MeshSynchronousDataStream
from cellier.models.scene.cameras import PerspectiveCamera
from cellier.models.scene.canvas import Canvas
from cellier.models.scene.dims_manager import CoordinateSystem, DimsManager
from cellier.models.scene.scene import Scene
from cellier.models.viewer import DataManager, SceneManager, ViewerModel
from cellier.models.visuals.mesh_visual import MeshPhongMaterial, MeshVisual
from cellier.viewer import Viewer

# the mesh data
vertices = np.array([[10, 10, 10], [30, 10, 20], [10, 20, 20]], dtype=np.float32)
faces = np.array([[0, 1, 2]], dtype=np.int32)

colors = np.array(
    [
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ],
    dtype=np.float32,
)

# make the mesh store
mesh_store = MeshMemoryStore(vertices=vertices, faces=faces)

# make the mesh stream
mesh_stream = MeshSynchronousDataStream(data_store=mesh_store, selectors=[])

# make the data manager
data = DataManager(stores=[mesh_store], streams=[mesh_stream])

# make the scene coordinate system
coordinate_system = CoordinateSystem(name="scene_0", axis_labels=["z", "y", "x"])
dims = DimsManager(
    coordinate_system=coordinate_system, displayed_dimensions=("z", "y", "x")
)

# make the mesh visual
mesh_material = MeshPhongMaterial()
mesh_visual = MeshVisual(
    name="mesh_visual", data_stream=mesh_stream, material=mesh_material
)

# make the canvas
camera = PerspectiveCamera(width=110, height=110)
canvas = Canvas(camera=camera)

canvas_dict = {canvas.id: "hello"}
print(canvas_dict)

# make the scene
scene = Scene(dims=dims, visuals=[mesh_visual], canvases=[canvas])
scene_manager = SceneManager(scenes=[scene])

scene_dict = {scene.id: "test"}
print(scene_dict)


viewer_model = ViewerModel(data=data, scenes=scene_manager)

print(viewer_model)


class Main(QtWidgets.QWidget):
    """Example widget with viewer."""

    def __init__(self):
        super().__init__(None)
        self.resize(640, 480)

        # Creat button and hook it up
        self._button = QtWidgets.QPushButton("Add a line", self)
        self._button.clicked.connect(self._on_button_click)

        # Create canvas, renderer and a scene object
        self._canvas = WgpuCanvas(parent=self)
        self._renderer = gfx.WgpuRenderer(self._canvas)
        self._scene = gfx.Scene()
        self._camera = gfx.OrthographicCamera(110, 110)
        self._controller = gfx.controllers.TrackballController(
            camera=self._camera, register_events=self._renderer
        )

        self._scene.add(
            gfx.Mesh(
                geometry=gfx.Geometry(indices=faces, positions=vertices),
                material=gfx.MeshPhongMaterial(),
            )
        )
        self._scene.add(gfx.AmbientLight())
        # self._scene.add(self._camera.add(gfx.DirectionalLight()))
        self._camera.show_object(self._scene)

        # Hook up the animate callback
        self._canvas.request_draw(self.animate)

        # make the viewer
        self.viewer = Viewer(model=viewer_model, widget_parent=self)

        for cam in self.viewer._render_manager.cameras.values():
            cam.show_pos((20, 15, 15), up=(0, 1, 0))

        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self._button)
        layout.addWidget(self._canvas)
        for canvas in self.viewer._canvas_widgets.values():
            # add the canvas widgets
            canvas.update()
            layout.addWidget(canvas)

    def _on_button_click(self):
        """Add lines when the button is clicked."""
        positions = [
            [random.uniform(-50, 50), random.uniform(-50, 50), 0] for i in range(8)
        ]
        line = gfx.Line(
            gfx.Geometry(positions=positions), gfx.LineMaterial(thickness=3)
        )
        self._scene.add(line)
        self._canvas.update()

    def animate(self):
        """Draw the canvas."""
        self._renderer.render(self._scene, self._camera)


app = QtWidgets.QApplication([])
m = Main()
m.show()


if __name__ == "__main__":
    app.exec()
