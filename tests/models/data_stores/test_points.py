import numpy as np

from cellier.models.data_stores import PointsMemoryStore
from cellier.models.scene import CoordinateSystem
from cellier.slicer import (
    AffineTransform,
    AxisAligned2DWorldSlice,
    AxisAligned3DWorldSlice,
    DataSliceRequest,
)


def test_point_memory_data_store_3d():
    """Test point data store accessing a 3D slice."""
    coordinates = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 10, 10, 10]])
    store = PointsMemoryStore(coordinates=coordinates)

    # make the world slice
    coordinate_system = CoordinateSystem(name="3d", axis_labels=("z", "y", "x"))
    world_slice = AxisAligned3DWorldSlice(
        displayed_dimensions=(1, 2, 3),
        point=(0, 9, 0, 0),
        margin_negative=(0, 1, 0, 0),
        margin_positive=(0, 1, 0, 0),
        world_coordinate_system=coordinate_system,
    )

    # make the transform
    transform = AffineTransform(matrix=np.eye(4))

    rendered_slice = store.get_slice(
        DataSliceRequest(
            world_slice=world_slice,
            visual_id="test",
            scene_id="test_scene",
            data_store_id="test",
            request_id="test",
            resolution_level=0,
            data_to_world_transform=transform,
        )
    )
    expected_points = np.array([[0, 0, 0], [10, 10, 10]])
    np.testing.assert_allclose(expected_points, rendered_slice.coordinates)


def test_point_memory_data_store_2d():
    """Test point data store accessing a 3D slice."""
    coordinates = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 10, 10, 10]])
    store = PointsMemoryStore(coordinates=coordinates)

    # make the world slice
    coordinate_system = CoordinateSystem(name="2d", axis_labels=("y", "x"))
    world_slice = AxisAligned2DWorldSlice(
        displayed_dimensions=(2, 3),
        point=(0, 9, 0, 0),
        margin_negative=(0, 1, 0, 0),
        margin_positive=(0, 1, 0, 0),
        world_coordinate_system=coordinate_system,
    )

    # make the transform
    transform = AffineTransform(matrix=np.eye(3))

    rendered_slice = store.get_slice(
        DataSliceRequest(
            visual_id="test",
            scene_id="test_scene",
            resolution_level=0,
            world_slice=world_slice,
            data_store_id="test",
            data_to_world_transform=transform,
            request_id="test",
        )
    )
    expected_points = np.array([[10, 10]])
    np.testing.assert_allclose(expected_points, rendered_slice.coordinates)
