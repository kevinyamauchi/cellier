"""Test the scene model."""

import json

import numpy as np
from pydantic_core import from_json

from cellier.models.data_stores import PointsMemoryStore
from cellier.models.scene import (
    Canvas,
    CoordinateSystem,
    DimsManager,
    PerspectiveCamera,
    Scene,
)
from cellier.models.visuals import PointsUniformMaterial, PointsVisual


def test_scene_model(tmp_path):
    """Test the serialization/deserialization of the Scene model."""

    coordinate_system = CoordinateSystem(name="default", axis_label=("x", "y", "z"))
    dims = DimsManager(
        coordinate_system=coordinate_system, displayed_dimensions=(0, 1, 2)
    )

    coordinates = np.array(
        [[10, 10, 10], [10, 10, 20], [10, 20, 20], [10, 20, 10]], dtype=np.float32
    )

    # make the points visual
    points_data = PointsMemoryStore(coordinates=coordinates)
    points_material = PointsUniformMaterial(
        size=1, color=(1, 0, 0, 1), size_coordinate_space="data"
    )

    points_visual = PointsVisual(
        name="test", data_store_id=points_data.id, material=points_material
    )

    # make the canvas
    canvas = Canvas(camera=PerspectiveCamera())

    # make the scene
    scene = Scene(dims=dims, visuals=[points_visual], canvases={canvas.id: canvas})

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
