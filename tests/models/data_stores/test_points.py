from uuid import uuid4

import numpy as np

from cellier.models.data_stores import PointsMemoryStore
from cellier.types import AxisAlignedSample, CoordinateSpace, TilingMethod


def test_point_memory_data_store_3d():
    """Test point data store accessing a 3D slice."""
    coordinates = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 10, 10, 10]])
    data_store = PointsMemoryStore(coordinates=coordinates)

    sample_request = AxisAlignedSample(
        space_type=CoordinateSpace.DATA,
        ordered_dims=(0, 1, 2, 3),
        n_displayed_dims=3,
        index_selection=(0, slice(None), slice(None), slice(None)),
    )

    data_requests = data_store.get_data_request(
        sample_request,
        tiling_method=TilingMethod.NONE,
        scene_id=uuid4().hex,
        visual_id=uuid4().hex,
    )
    assert len(data_requests) == 1

    data_response = data_store.get_data(data_requests[0])

    expected_points = np.array([[0, 0, 0], [10, 10, 10]])
    np.testing.assert_allclose(expected_points, data_response.data)


def test_point_memory_data_store_2d():
    """Test point data store accessing a 2D slice with margins."""
    coordinates = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 10, 10, 10]])
    data_store = PointsMemoryStore(coordinates=coordinates)

    sample_request = AxisAlignedSample(
        space_type=CoordinateSpace.DATA,
        ordered_dims=(0, 1, 2, 3),
        n_displayed_dims=2,
        index_selection=(0, slice(9, 11), slice(None), slice(None)),
    )

    data_requests = data_store.get_data_request(
        sample_request,
        tiling_method=TilingMethod.NONE,
        scene_id=uuid4().hex,
        visual_id=uuid4().hex,
    )
    assert len(data_requests) == 1

    data_response = data_store.get_data(data_requests[0])

    expected_points = np.array([[10, 10]])
    np.testing.assert_allclose(expected_points, data_response.data)
