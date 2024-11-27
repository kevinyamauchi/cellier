import numpy as np

from cellier.util.geometry import (
    frustum_planes_from_corners,
    near_far_plane_edge_lengths,
    points_in_frustum,
)


def test_points_in_frustum():
    """Test determining if a point is in a frustum ."""
    points = np.array(
        [
            [0, 0, 0],  # Inside
            [5, 5, 5],  # Outside
            [1, 1, 1],  # Inside
            [-1, -1, -2],  # Outside
        ]
    )

    # Define frustum planes
    # Each row is [a, b, c, d]
    planes = np.array(
        [
            [1, 0, 0, 1],  # Left plane
            [-1, 0, 0, 1],  # Right plane
            [0, 1, 0, 1],  # Bottom plane
            [0, -1, 0, 1],  # Top plane
            [0, 0, 1, 1],  # Near plane
            [0, 0, -1, 10],  # Far plane
        ]
    )
    points_mask = points_in_frustum(points=points, planes=planes)
    assert np.all(points_mask == np.array([True, False, True, False]))


def test_frustum_planes_from_corners():
    """Test determining the plane parameters from frustum corners."""

    corners = np.array(
        [
            [
                [-1, -1, -1],
                [-1, 1, -1],
                [1, 1, -1],
                [1, -1, -1],
            ],
            [
                [-1, -1, 10],
                [-1, 1, 10],
                [1, 1, 10],
                [1, -1, 10],
            ],
        ]
    )

    planes = frustum_planes_from_corners(corners)
    expected_planes = np.array(
        [
            [0, 0, 1, 1],  # Near plane
            [0, 0, -1, 10],  # Far plane
            [0, 1, 0, 1],  # Left plane
            [0, -1, 0, 1],  # Right plane
            [-1, 0, 0, 1],  # Top plane
            [1, 0, 0, 1],  # Bottom plane
        ]
    )
    np.testing.assert_allclose(planes, expected_planes)


def test_near_far_plane_edge_lengths():
    """Test calculated the edge lengths of the near and far plane."""

    corners = np.array(
        [
            [
                [-1, -2, -1],
                [1, -2, -1],
                [1, 2, -1],
                [-1, 2, -1],
            ],
            [
                [-2, -4, 10],
                [2, -4, 10],
                [2, 4, 10],
                [-2, 4, 10],
            ],
        ]
    )
    edge_lengths = near_far_plane_edge_lengths(corners=corners)

    expected_edge_lengths = np.array([[2, 4, 2, 4], [4, 8, 4, 8]])
    np.testing.assert_allclose(edge_lengths, expected_edge_lengths)
