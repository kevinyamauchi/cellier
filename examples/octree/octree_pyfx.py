"""Example showing progressive loading of volume data."""

import numpy as np
import pygfx as gfx
from qtpy import QtWidgets
from wgpu.gui.qt import WgpuCanvas

from cellier.util.chunk import ChunkData3D
from cellier.util.geometry import (
    frustum_edges_from_corners,
    frustum_planes_from_corners,
)


class Main(QtWidgets.QWidget):
    """Main window."""

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
        self._controller = gfx.OrbitController(
            self._camera, register_events=self._renderer
        )

        # make the scene
        volume_image = np.zeros((100, 100, 100), dtype=np.float32)
        tex = gfx.Texture(volume_image, dim=3)
        self.vol = gfx.Volume(
            gfx.Geometry(grid=tex),
            gfx.VolumeRayMaterial(clim=(0, 1), map=gfx.cm.cividis),
        )
        self._scene.add(self.vol)

        # make the grid of points
        self._chunk_data = ChunkData3D(
            array_shape=(1000, 1000, 1000), chunk_shape=(25, 25, 25)
        )

        self._grid_points = self._chunk_data.chunk_centers.astype(np.float32)
        self._points = gfx.Points(
            gfx.Geometry(positions=self._grid_points),
            gfx.PointsMaterial(size=10, color=(0, 1, 0.5, 1.0)),
        )
        self._scene.add(self._points)

        self._camera.show_object(self.vol, view_dir=(0, 1, 0), up=(0, 0, 1))

        # Hook up the animate callback
        self._canvas.request_draw(self.animate)

        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self._button)
        layout.addWidget(self._canvas)

    def _on_button_click(self):
        texture = self.vol.geometry.grid
        texture.data[0:50, :, :] = 0.25
        texture.update_range((0, 0, 0), (100, 100, 50))
        frustum_edges = frustum_edges_from_corners(self._camera.frustum)

        line_coordinates = frustum_edges.reshape(24, 3)
        line = gfx.Line(
            gfx.Geometry(positions=line_coordinates),
            gfx.LineSegmentMaterial(thickness=3),
        )
        self._scene.add(line)

        frustum_planes = frustum_planes_from_corners(self._camera.frustum)
        chunk_mask = self._chunk_data.chunks_in_frustum(planes=frustum_planes)
        self._points.geometry = gfx.Geometry(
            positions=self._chunk_data.chunk_centers[chunk_mask].astype(np.float32)
        )

        self._renderer.request_draw()

    def animate(self):
        """Run the render loop."""
        self._renderer.render(self._scene, self._camera)


app = QtWidgets.QApplication([])
m = Main()
m.show()


if __name__ == "__main__":
    app.exec()
