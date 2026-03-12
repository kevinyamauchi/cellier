"""Shared pytest fixtures for chunked_image tests."""

import numpy as np
import pytest

from cellier.transform import AffineTransform
from cellier.utils.chunked_image import ScaleLevelModel
from cellier.utils.chunked_image._chunk_culler import ChunkCuller
from cellier.utils.chunked_image._chunk_selection import (
    AxisAlignedTexturePositioning,
    ChunkSelector,
    TextureBoundsFiltering,
)
from cellier.utils.chunked_image._data_classes import (
    TextureConfiguration,
)

# ---------------------------------------------------------------------------
# Scale level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scale_level_identity():
    """128^3 volume, 32^3 chunks, identity transform -> 4x4x4 = 64 chunks."""
    return ScaleLevelModel(
        shape=(128, 128, 128),
        chunk_shape=(32, 32, 32),
        transform=AffineTransform.from_scale_and_translation(
            (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
        ),
    )


@pytest.fixture
def scale_level_2x():
    """64^3 downsampled volume, 16^3 chunks, 2x downscale -> 4x4x4 = 64 chunks."""
    return ScaleLevelModel(
        shape=(64, 64, 64),
        chunk_shape=(16, 16, 16),
        transform=AffineTransform.from_scale_and_translation(
            (2.0, 2.0, 2.0), (0.0, 0.0, 0.0)
        ),
    )


@pytest.fixture
def identity_world_transform():
    """Identity world transform (world == scale_0)."""
    return AffineTransform.from_translation((0.0, 0.0, 0.0))


@pytest.fixture
def scale2_world_transform():
    """World transform that doubles all coordinates: world = 2 * scale_0."""
    return AffineTransform.from_scale_and_translation((2.0, 2.0, 2.0), (0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Strategy / selector fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chunk_culler():
    return ChunkCuller()


@pytest.fixture
def chunk_selector():
    return ChunkSelector(
        positioning_strategy=AxisAlignedTexturePositioning(),
        filtering_strategy=TextureBoundsFiltering(),
    )


@pytest.fixture
def texture_config_64():
    return TextureConfiguration(texture_width=64)


# ---------------------------------------------------------------------------
# Helper: build axis-aligned frustum corners in (z, y, x) space
# ---------------------------------------------------------------------------


def make_box_frustum(
    z_near: float,
    z_far: float,
    y_min: float,
    y_max: float,
    x_min: float,
    x_max: float,
) -> np.ndarray:
    """Return (2, 4, 3) frustum corners for a camera looking along +z.

    Corner order per plane: (left_bottom, right_bottom, right_top, left_top).
    Coordinates are in (z, y, x) order.

    The resulting frustum encloses the region
    z in [z_near, z_far], y in [y_min, y_max], x in [x_min, x_max].

    Plane normals all point inward, which is the convention expected by
    ``frustum_planes_from_corners`` / ``points_in_frustum``.
    """
    return np.array(
        [
            # near plane (z = z_near)
            [
                [z_near, y_min, x_min],  # left_bottom
                [z_near, y_min, x_max],  # right_bottom
                [z_near, y_max, x_max],  # right_top
                [z_near, y_max, x_min],  # left_top
            ],
            # far plane (z = z_far)
            [
                [z_far, y_min, x_min],
                [z_far, y_min, x_max],
                [z_far, y_max, x_max],
                [z_far, y_max, x_min],
            ],
        ],
        dtype=np.float64,
    )
