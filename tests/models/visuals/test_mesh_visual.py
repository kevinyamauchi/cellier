"""Tests for the mesh visual models."""

import json

import numpy as np
from pydantic_core import from_json

from cellier.models.data_stores.mesh import MeshMemoryStore
from cellier.models.data_streams.mesh import MeshSynchronousDataStream
from cellier.models.nodes.mesh_visual import MeshNode, MeshPhongMaterial


def test_mesh_visual(tmp_path):
    """Test serialization/deserialization of the Mesh Visual model."""
    vertices = np.array(
        [[10, 10, 10], [10, 10, 20], [10, 20, 20], [10, 20, 10]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.float32)

    mesh = MeshMemoryStore(vertices=vertices, faces=faces)
    mesh_stream = MeshSynchronousDataStream(data_store_id=mesh.id, selectors=[])
    mesh_material = MeshPhongMaterial()

    mesh_visual = MeshNode(
        name="test", data_stream_id=mesh_stream.id, material=mesh_material
    )

    output_path = tmp_path / "test.json"
    with open(output_path, "w") as f:
        # serialize the model
        json.dump(mesh_visual.model_dump(), f)

    # deserialize
    with open(output_path, "rb") as f:
        deserialized_visual = MeshNode.model_validate(
            from_json(f.read(), allow_partial=False)
        )

    assert deserialized_visual.material == mesh_visual.material

    # test the mesh data is correct
    assert mesh_stream.id == deserialized_visual.data_stream_id
    assert mesh_material == deserialized_visual.material
