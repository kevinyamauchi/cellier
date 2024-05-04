"""Test the scene model."""

import json

import numpy as np
from pydantic_core import from_json

from cellier.models.data_stores.mesh import MeshMemoryStore
from cellier.models.data_streams.mesh import MeshSynchronousDataStream
from cellier.models.scene import (
    Canvas,
    CoordinateSystem,
    DimsManager,
    PerspectiveCamera,
    Scene,
)
from cellier.models.visuals.mesh_visual import MeshPhongMaterial, MeshVisual


def test_scene_model(tmp_path):
    """Test the serialization/deserialization of the Scene model."""

    coordinate_system = CoordinateSystem(name="default", axis_label=("x", "y", "z"))
    dims = DimsManager(
        coordinate_system=coordinate_system, displayed_dimensions=(0, 1, 2)
    )

    vertices = np.array(
        [[10, 10, 10], [10, 10, 20], [10, 20, 20], [10, 20, 10]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.float32)

    # make the mesh visual
    mesh = MeshMemoryStore(vertices=vertices, faces=faces)
    mesh_stream = MeshSynchronousDataStream(data_store_id=mesh.id, selectors=[])
    mesh_material = MeshPhongMaterial()

    mesh_visual = MeshVisual(
        name="test", data_stream_id=mesh_stream.id, material=mesh_material
    )

    # make the canvas
    canvas = Canvas(camera=PerspectiveCamera())

    # make the scene
    scene = Scene(dims=dims, visuals=[mesh_visual], canvases=[canvas])

    output_path = tmp_path / "test.json"
    with open(output_path, "w") as f:
        # serialize the model
        json.dump(scene.model_dump(), f)

    # deserialize
    with open(output_path, "rb") as f:
        deserialized_scene = Scene.model_validate(
            from_json(f.read(), allow_partial=False)
        )

    assert deserialized_scene.dims == dims
