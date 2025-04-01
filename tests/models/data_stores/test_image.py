from uuid import uuid4

import numpy as np
import pytest

from cellier.models.data_stores import ImageMemoryStore
from cellier.types import AxisAlignedSample, CoordinateSpace, TilingMethod


def test_axis_aligned_sample_memory_store_2d():
    """Test sampling an Image memory store axis aligned."""
    image = np.random.random((10, 10, 10))

    data_store = ImageMemoryStore(data=image)

    sample_request = AxisAlignedSample(
        space_type=CoordinateSpace.DATA,
        ordered_dims=(0, 1, 2),
        n_displayed_dims=2,
        index_selection=(5, slice(None), slice(None)),
        tiling_method=TilingMethod.NONE,
    )

    data_requests = data_store.get_data_request(
        sample_request,
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

    sample_request = AxisAlignedSample(
        space_type=CoordinateSpace.DATA,
        ordered_dims=(0, 1, 2),
        n_displayed_dims=3,
        index_selection=(slice(5, 8), slice(6, 9), slice(7, 10)),
        tiling_method=TilingMethod.NONE,
    )
    data_requests = data_store.get_data_request(
        sample_request,
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


def test_tiling_memory_store():
    """Tiling is not implemented for the ImageMemoryStore."""
    image = np.random.random((10, 10, 10))

    data_store = ImageMemoryStore(data=image)

    sample_request = AxisAlignedSample(
        space_type=CoordinateSpace.DATA,
        ordered_dims=(0, 1, 2),
        n_displayed_dims=2,
        index_selection=(slice(5, 6), slice(None), slice(None)),
        tiling_method=TilingMethod.LOGICAL_PIXEL,
    )

    with pytest.raises(NotImplementedError):
        _ = data_store.get_data_request(
            sample_request,
            visual_id=uuid4().hex,
            scene_id=uuid4().hex,
        )
