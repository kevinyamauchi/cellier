import json

import numpy as np
from pydantic_core import from_json

from cellier.models.data_stores.mesh import MeshMemoryStore
from cellier.models.data_streams.mesh import MeshSynchronousDataStream


def test_mesh_synchronous_data_stream(tmp_path):
    """Test serialization/deserialization of the MeshSynchronousDataStream."""
    vertices = np.array(
        [[10, 10, 10], [10, 10, 20], [10, 20, 20], [10, 20, 10]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.float32)

    mesh = MeshMemoryStore(vertices=vertices, faces=faces)
    mesh_stream = MeshSynchronousDataStream(data_store=mesh, selectors=[])

    output_path = tmp_path / "test.json"
    with open(output_path, "w") as f:
        # serialize the model
        json.dump(mesh_stream.model_dump(), f)

    # deserialize
    with open(output_path, "rb") as f:
        deserialized_stream = MeshSynchronousDataStream.model_validate(
            from_json(f.read(), allow_partial=False)
        )

    # the that the values are correct
    np.testing.assert_allclose(vertices, deserialized_stream.data_store.vertices)
    np.testing.assert_allclose(faces, deserialized_stream.data_store.faces)
