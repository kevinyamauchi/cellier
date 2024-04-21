"""Simple example of making the viewer via the Python API."""

import numpy as np

from cellier.models.data_streams.mesh_data import MeshDataSource
from cellier.models.scene.cameras import PerspectiveCamera
from cellier.viewer import Viewer

vertices = np.array([[10, 10, 10], [10, 10, 20], [10, 20, 20]], dtype=np.float32)
faces = np.array([[0, 1, 2]], dtype=np.float32)

colors = np.array(
    [
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ],
    dtype=np.float32,
)

# make the data_streams source
mesh_data = MeshDataSource(vertices=vertices, faces=faces, colors=colors)

# make the material


# make the camera
camera = PerspectiveCamera()


# make the viewer
viewer = Viewer(camera=camera)

# add the mesh
viewer.add_layer()
