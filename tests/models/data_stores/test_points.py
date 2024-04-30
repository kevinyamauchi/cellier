import numpy as np

from cellier.models.data_stores.points import PointDataStoreSlice, PointsMemoryStore


def test_point_memory_data_store_3d():
    """Test point data store accessing a 3D slice."""
    coordinates = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 10, 10, 10]])
    store = PointsMemoryStore(coordinates=coordinates)

    rendered_slice = store.get_slice(
        PointDataStoreSlice(
            visual_id="test",
            scene_id="test_scene",
            resolution_level=0,
            displayed_dimensions=(1, 2, 3),
            point=(0, 0, 0, 0),
            margin_negative=(0, 0, 0, 0),
            margin_positive=(0, 0, 0, 0),
        )
    )
    expected_points = np.array([[0, 0, 0], [10, 10, 10]])
    np.testing.assert_allclose(expected_points, rendered_slice.coordinates)


def test_point_memory_data_store_2d():
    """Test point data store accessing a 3D slice."""
    coordinates = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 10, 10, 10]])
    store = PointsMemoryStore(coordinates=coordinates)

    rendered_slice = store.get_slice(
        PointDataStoreSlice(
            visual_id="test",
            scene_id="test_scene",
            resolution_level=0,
            displayed_dimensions=(2, 3),
            point=(0, 9, 0, 0),
            margin_negative=(0, 1, 0, 0),
            margin_positive=(0, 1, 0, 0),
        )
    )
    expected_points = np.array([[10, 10]])
    np.testing.assert_allclose(expected_points, rendered_slice.coordinates)
