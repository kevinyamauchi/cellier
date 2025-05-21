from uuid import uuid4

import numpy as np
import pytest

from cellier.models.data_stores import LinesMemoryStore
from cellier.types import (
    AxisAlignedSelectedRegion,
    CoordinateSpace,
    PlaneSelectedRegion,
    TilingMethod,
)


def test_lines_memory_data_store_2d():
    """Test lines data store accessing a thick 2D axis-aligned slice."""
    coordinates = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 10, 10],
            [1, 1, 1, 1],
            [1, 1, 11, 11],
            [0, 10, 10, 10],
            [0, 10, 20, 20],
        ]
    )
    data_store = LinesMemoryStore(coordinates=coordinates)

    sample_request = AxisAlignedSelectedRegion(
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

    expected_points = np.array(
        [
            [10, 10],
            [20, 20],
        ]
    )
    np.testing.assert_allclose(expected_points, data_response.data)


def test_lines_memory_data_store_3d():
    """Test lines data store accessing a 3D slice."""
    coordinates = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 10, 10],
            [1, 1, 1, 1],
            [1, 1, 11, 11],
            [0, 10, 10, 10],
            [0, 10, 20, 20],
        ]
    )
    data_store = LinesMemoryStore(coordinates=coordinates)

    sample_request = AxisAlignedSelectedRegion(
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

    expected_points = np.array(
        [
            [0, 0, 0],
            [0, 10, 10],
            [10, 10, 10],
            [10, 20, 20],
        ]
    )
    np.testing.assert_allclose(expected_points, data_response.data)


def test_lines_memory_data_store_rolled_axes_3d():
    """Test lines data store accessing a 3D slice."""
    coordinates = np.array(
        [
            [1, 2, 3, 0],
            [4, 5, 6, 0],
            [1, 1, 1, 1],
            [1, 1, 11, 11],
            [0, 10, 10, 10],
            [0, 10, 20, 20],
        ]
    )
    data_store = LinesMemoryStore(coordinates=coordinates)

    sample_request = AxisAlignedSelectedRegion(
        space_type=CoordinateSpace.DATA,
        ordered_dims=(3, 0, 1, 2),
        n_displayed_dims=3,
        index_selection=(slice(None, None), slice(None), slice(None), 0),
    )

    data_requests = data_store.get_data_request(
        sample_request,
        tiling_method=TilingMethod.NONE,
        scene_id=uuid4().hex,
        visual_id=uuid4().hex,
    )
    assert len(data_requests) == 1

    data_response = data_store.get_data(data_requests[0])

    expected_points = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    np.testing.assert_allclose(expected_points, data_response.data)


def test_lines_memory_data_plane_sample():
    """Currently, plane sampling is not implemented for lines."""
    coordinates = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 10, 10],
            [1, 1, 1, 1],
            [1, 1, 11, 11],
            [0, 10, 10, 10],
            [0, 10, 20, 20],
        ]
    )
    data_store = LinesMemoryStore(coordinates=coordinates)

    plane_transform = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]
    )
    sample_request = PlaneSelectedRegion(
        space_type=CoordinateSpace.DATA,
        ordered_dims=(0, 1, 2, 3),
        n_displayed_dims=3,
        index_selection=(2, slice(None), slice(None), slice(None)),
        point=np.array([0, 5, 0]),
        plane_transform=plane_transform,
        extents=(10, 10),
    )

    data_requests = data_store.get_data_request(
        sample_request,
        tiling_method=TilingMethod.NONE,
        scene_id=uuid4().hex,
        visual_id=uuid4().hex,
    )
    assert len(data_requests) == 1

    with pytest.raises(NotImplementedError):
        _ = data_store.get_data(data_requests[0])


def test_lines_memory_data_tiling():
    """Currently, tiling is not implemented for lines."""
    coordinates = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 10, 10],
            [1, 1, 1, 1],
            [1, 1, 11, 11],
            [0, 10, 10, 10],
            [0, 10, 20, 20],
        ]
    )
    data_store = LinesMemoryStore(coordinates=coordinates)

    sample_request = AxisAlignedSelectedRegion(
        space_type=CoordinateSpace.DATA,
        ordered_dims=(0, 1, 2, 3),
        n_displayed_dims=2,
        index_selection=(0, slice(9, 11), slice(None), slice(None)),
    )

    with pytest.raises(NotImplementedError):
        _ = data_store.get_data_request(
            sample_request,
            tiling_method=TilingMethod.LOGICAL_PIXEL,
            scene_id=uuid4().hex,
            visual_id=uuid4().hex,
        )


def test_point_memory_data_bad_selected_region():
    """An invalid selected region should raise a TypeError."""
    coordinates = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 10, 10],
            [1, 1, 1, 1],
            [1, 1, 11, 11],
            [0, 10, 10, 10],
            [0, 10, 20, 20],
        ]
    )
    data_store = LinesMemoryStore(coordinates=coordinates)

    with pytest.raises(TypeError):
        _ = data_store.get_data_request(
            "data please",
            tiling_method=TilingMethod.NONE,
            scene_id=uuid4().hex,
            visual_id=uuid4().hex,
        )


def test_point_memory_data_bad_data_request():
    """An invalid selected region should raise a TypeError."""
    coordinates = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 10, 10],
            [1, 1, 1, 1],
            [1, 1, 11, 11],
            [0, 10, 10, 10],
            [0, 10, 20, 20],
        ]
    )
    data_store = LinesMemoryStore(coordinates=coordinates)

    with pytest.raises(TypeError):
        _ = data_store.get_data_request("data please")
