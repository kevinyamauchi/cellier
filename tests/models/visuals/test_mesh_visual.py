"""Tests for the mesh visual models."""

import json

import numpy as np
import pytest
from pydantic_core import from_json


@pytest.mark.xfail(reason="MeshVisual is not implemented yet", raises=ImportError)
def test_mesh_visual(tmp_path):
    """Test serialization/deserialization of the Mesh Visual model."""
    from cellier.models.data_stores import MeshMemoryStore
    from cellier.models.visuals import MeshPhongMaterial, MeshVisual

    vertices = np.array(
        [[10, 10, 10], [10, 10, 20], [10, 20, 20], [10, 20, 10]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.float32)

    mesh_store = MeshMemoryStore(vertices=vertices, faces=faces)
    mesh_material = MeshPhongMaterial()

    mesh_visual = MeshVisual(
        name="test", data_store_id=mesh_store.id, material=mesh_material
    )

    output_path = tmp_path / "test.json"
    with open(output_path, "w") as f:
        # serialize the model
        json.dump(mesh_visual.model_dump(), f)

    # deserialize
    with open(output_path, "rb") as f:
        deserialized_visual = MeshVisual.model_validate(
            from_json(f.read(), allow_partial=False)
        )

    assert deserialized_visual.material == mesh_visual.material

    # test the mesh data is correct
    assert mesh_store.id == deserialized_visual.data_store_id
    assert mesh_material == deserialized_visual.material
