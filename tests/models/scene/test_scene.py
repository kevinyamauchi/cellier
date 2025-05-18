"""Test the scene model."""

import numpy as np
from pydantic_core import from_json

from cellier.models.data_stores import PointsMemoryStore
from cellier.models.scene import (
    AxisAlignedRegionSelector,
    Canvas,
    CoordinateSystem,
    DimsManager,
    OrbitCameraController,
    PerspectiveCamera,
    RangeTuple,
    Scene,
)
from cellier.models.visuals import PointsUniformMaterial, PointsVisual
from cellier.types import CoordinateSpace


def test_scene_model(tmp_path):
    """Test the serialization/deserialization of the Scene model."""

    data_range = (
        RangeTuple(start=0, stop=250, step=1),
        RangeTuple(start=0, stop=250, step=1),
        RangeTuple(start=0, stop=250, step=1),
    )
    coordinate_system_3d = CoordinateSystem(
        name="scene_3d", axis_labels=("z", "y", "x")
    )
    selection = AxisAlignedRegionSelector(
        space_type=CoordinateSpace.WORLD,
        ordered_dims=(0, 1, 2),
        n_displayed_dims=3,
        index_selection=(0, slice(0, 10, 1), slice(None, None, None)),
    )
    dims = DimsManager(
        range=data_range,
        coordinate_system=coordinate_system_3d,
        selection=selection,
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
    canvas = Canvas(
        camera=PerspectiveCamera(controller=OrbitCameraController(enabled=True))
    )

    # make the scene
    scene = Scene(dims=dims, visuals=[points_visual], canvases={canvas.id: canvas})

    output_path = tmp_path / "test.json"
    with open(output_path, "w") as f:
        # serialize the model
        f.write(scene.model_dump_json())

    # deserialize
    with open(output_path, "rb") as f:
        deserialized_scene = Scene.model_validate(
            from_json(f.read(), allow_partial=False)
        )

    assert deserialized_scene.dims == dims
