import json

import numpy as np
import pytest
from pydantic_core import from_json


@pytest.mark.xfail(reason="meshes not implemented", raises=ImportError)
def test_mesh_data_store():
    from cellier.models import MeshMemoryStore

    vertices = np.array(
        [[10, 10, 10], [10, 10, 20], [10, 20, 20], [10, 20, 10]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.float32)

    mesh = MeshMemoryStore(vertices=vertices, faces=faces)
    np.testing.assert_allclose(vertices, mesh.vertices)
    np.testing.assert_allclose(faces, mesh.faces)


@pytest.mark.xfail(reason="meshes not implemented", raises=ImportError)
def test_mesh_memory_store_serialization(tmp_path):
    """test serialization and deserialization of MeshMemoryStore."""
    from cellier.models import MeshMemoryStore

    vertices = np.array(
        [[10, 10, 10], [10, 10, 20], [10, 20, 20], [10, 20, 10]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.float32)

    mesh = MeshMemoryStore(vertices=vertices, faces=faces)

    output_path = tmp_path / "test.json"
    with open(output_path, "w") as f:
        # serialize the model
        json.dump(mesh.model_dump(), f)

    # deserialize
    with open(output_path, "rb") as f:
        deserialized_mesh = MeshMemoryStore.model_validate(
            from_json(f.read(), allow_partial=False)
        )

    # the that the values are correct
    np.testing.assert_allclose(vertices, deserialized_mesh.vertices)
    np.testing.assert_allclose(faces, deserialized_mesh.faces)
