from uuid import uuid4

import numpy as np
import pytest

from cellier.models.data_stores import ImageMemoryStore
from cellier.types import (
    AxisAlignedSelectedRegion,
    CoordinateSpace,
    PlaneSelectedRegion,
    TilingMethod,
)


def test_axis_aligned_sample_memory_store_2d():
    """Test sampling an Image memory store axis aligned."""
    image = np.random.random((10, 10, 10))

    data_store = ImageMemoryStore(data=image)

    sample_request = AxisAlignedSelectedRegion(
        space_type=CoordinateSpace.DATA,
        ordered_dims=(0, 1, 2),
        n_displayed_dims=2,
        index_selection=(5, slice(None), slice(None)),
    )

    data_requests = data_store.get_data_request(
        sample_request,
        tiling_method=TilingMethod.NONE,
        scene_id=uuid4().hex,
        visual_id=uuid4().hex,
    )
    assert len(data_requests) == 1

    data_response = data_store.get_data(data_requests[0])
    np.testing.assert_allclose(
        data_response.data,
        image[5, ...],
    )
    assert data_response.id == data_requests[0].id
    assert data_response.resolution_level == 0
    assert data_response.min_corner_rendered == (0, 0)


def test_axis_aligned_sample_memory_store_3d():
    """Test a 3D axis aligned sample from a memory store."""
    image = np.zeros((10, 10, 10))
    image[5:8, 6:9, 7:10] = 1

    data_store = ImageMemoryStore(data=image)

    sample_request = AxisAlignedSelectedRegion(
        space_type=CoordinateSpace.DATA,
        ordered_dims=(0, 1, 2),
        n_displayed_dims=3,
        index_selection=(slice(5, 8), slice(6, 9), slice(7, 10)),
    )
    data_requests = data_store.get_data_request(
        sample_request,
        tiling_method=TilingMethod.NONE,
        scene_id=uuid4().hex,
        visual_id=uuid4().hex,
    )
    assert len(data_requests) == 1

    data_response = data_store.get_data(data_requests[0])
    np.testing.assert_allclose(
        data_response.data,
        np.ones((3, 3, 3)),
    )
    assert data_response.id == data_requests[0].id
    assert data_response.resolution_level == 0
    assert data_response.min_corner_rendered == (5, 6, 7)


def test_axis_aligned_sample_memory_store_rolled_axes():
    """Test sampling with rolled axes."""
    image = np.array(
        [
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
            ],
            [
                [9, 10, 11],
                [12, 13, 14],
                [15, 16, 17],
            ],
            [
                [18, 19, 20],
                [21, 22, 23],
                [24, 25, 26],
            ],
        ]
    )

    data_store = ImageMemoryStore(data=image)
    ordered_dims = (1, 2, 0)
    expected_result = np.array(
        [
            [0, 9, 18],
            [1, 10, 19],
            [2, 11, 20],
        ]
    )

    sample_request = AxisAlignedSelectedRegion(
        space_type=CoordinateSpace.DATA,
        ordered_dims=ordered_dims,
        n_displayed_dims=2,
        index_selection=(slice(None, None), 0, slice(None, None)),
    )
    data_requests = data_store.get_data_request(
        sample_request,
        tiling_method=TilingMethod.NONE,
        scene_id=uuid4().hex,
        visual_id=uuid4().hex,
    )
    assert len(data_requests) == 1

    data_response = data_store.get_data(data_requests[0])

    # check the result
    np.testing.assert_allclose(
        data_response.data,
        expected_result,
    )


def test_plane_sample_memory_store_3d():
    """Test a 2D plane sample from a 3D image memory store."""
    image = np.zeros((10, 10, 10))

    plane_2d = np.zeros((10, 10))
    plane_2d[0:5, :] = 1
    plane_2d[5:, :] = 2
    image[:, 5, :] = plane_2d

    data_store = ImageMemoryStore(data=image)

    plane_transform = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]
    )
    sample_request = PlaneSelectedRegion(
        space_type=CoordinateSpace.DATA,
        ordered_dims=(0, 1, 2),
        n_displayed_dims=3,
        index_selection=(slice(None), slice(None), slice(None)),
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

    data_response = data_store.get_data(data_requests[0])

    np.testing.assert_allclose(
        data_response.data,
        plane_2d,
    )


def test_plane_sample_memory_store_4d():
    """Test a 2D plane sample from a 4D image memory store."""
    image = np.zeros((10, 10, 10, 10))

    plane_2d = np.zeros((10, 10))
    plane_2d[0:5, :] = 1
    plane_2d[5:, :] = 2
    image[2, :, 5, :] = plane_2d

    data_store = ImageMemoryStore(data=image)

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

    data_response = data_store.get_data(data_requests[0])

    np.testing.assert_allclose(
        data_response.data,
        plane_2d,
    )


def test_tiling_memory_store():
    """Tiling is not implemented for the ImageMemoryStore."""
    image = np.random.random((10, 10, 10))

    data_store = ImageMemoryStore(data=image)

    sample_request = AxisAlignedSelectedRegion(
        space_type=CoordinateSpace.DATA,
        ordered_dims=(0, 1, 2),
        n_displayed_dims=2,
        index_selection=(slice(5, 6), slice(None), slice(None)),
    )

    with pytest.raises(NotImplementedError):
        _ = data_store.get_data_request(
            sample_request,
            tiling_method=TilingMethod.LOGICAL_PIXEL,
            visual_id=uuid4().hex,
            scene_id=uuid4().hex,
        )


def test_image_memory_data_bad_selected_region():
    """An invalid selected region should raise a TypeError."""
    image = np.zeros((10, 10, 10))
    data_store = ImageMemoryStore(data=image)

    with pytest.raises(TypeError):
        _ = data_store.get_data_request(
            "data please",
            tiling_method=TilingMethod.NONE,
            scene_id=uuid4().hex,
            visual_id=uuid4().hex,
        )


def test_image_memory_data_bad_data_request():
    """An invalid selected region should raise a TypeError."""
    image = np.zeros((10, 10, 10))
    data_store = ImageMemoryStore(data=image)

    with pytest.raises(TypeError):
        _ = data_store.get_data_request("data please")
