import numpy as np

from cellier.models.data_streams.mesh_data import MeshDataSource


def test_mesh_data_sources():
    vertices = np.array(
        [[10, 10, 10], [10, 10, 20], [10, 20, 20], [10, 20, 10]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.float32)
    colors = np.ones((len(vertices), 4), dtype=np.float32)

    mesh = MeshDataSource(vertices, faces, colors)
    np.testing.assert_allclose(vertices, mesh.vertices)
    np.testing.assert_allclose(faces, mesh.faces)
    np.testing.assert_allclose(colors, mesh.colors)
