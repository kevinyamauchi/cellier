import numpy as np

from cellier.util.geometry import (
    frustum_planes_from_corners,
    generate_2d_grid,
    generate_3d_grid,
    near_far_plane_edge_lengths,
    plane_intersects_aabb,
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


def test_plane_intersects_aabb_3d():
    """Test finding the intersection of a plane and an AABB for 3D."""

    # plane inside of the box
    assert plane_intersects_aabb(
        plane_point=np.array([10, 10, 10]),
        plane_normal=np.array([1, 0, 0]),
        bounding_box_min=np.array([0, 0, 0]),
        bounding_box_max=np.array([15, 15, 15]),
    )

    # includes plane on the edge of the box
    assert plane_intersects_aabb(
        plane_point=np.array([10, 10, 10]),
        plane_normal=np.array([1, 0, 0]),
        bounding_box_min=np.array([10, 10, 10]),
        bounding_box_max=np.array([15, 15, 15]),
    )

    assert plane_intersects_aabb(
        plane_point=np.array([10, 10, 10]),
        plane_normal=np.array([1, 0, 0]),
        bounding_box_min=np.array([10, 12, 12]),
        bounding_box_max=np.array([10, 15, 15]),
    )

    # plane outside of the box
    assert not plane_intersects_aabb(
        plane_point=np.array([10, 10, 10]),
        plane_normal=np.array([1, 0, 0]),
        bounding_box_min=np.array([11, 11, 11]),
        bounding_box_max=np.array([15, 15, 15]),
    )


def test_plane_intersects_aabb_4d():
    """Test finding the intersection of a plane and an AABB for 3D."""
    # plane inside of the box
    assert plane_intersects_aabb(
        plane_point=np.array([10, 10, 10, 10]),
        plane_normal=np.array([10, 1, 0, 0]),
        bounding_box_min=np.array([10, 0, 0, 0]),
        bounding_box_max=np.array([10, 15, 15, 15]),
    )

    # includes plane on the edge
    assert plane_intersects_aabb(
        plane_point=np.array([10, 10, 10, 10]),
        plane_normal=np.array([10, 1, 0, 0]),
        bounding_box_min=np.array([10, 10, 10, 10]),
        bounding_box_max=np.array([10, 15, 15, 15]),
    )

    # plane outside of the box
    assert not plane_intersects_aabb(
        plane_point=np.array([10, 10, 10, 10]),
        plane_normal=np.array([10, 1, 0, 0]),
        bounding_box_min=np.array([10, 11, 11, 11]),
        bounding_box_max=np.array([10, 15, 15, 15]),
    )


def test_plane_intersects_aabb_5d():
    """Test finding the intersection of a plane and an AABB for 3D."""
    # plane inside of the box
    assert plane_intersects_aabb(
        plane_point=np.array([10, 10, 10, 10, 10]),
        plane_normal=np.array([10, 1, 0, 10, 0]),
        bounding_box_min=np.array([10, 0, 0, 10, 0]),
        bounding_box_max=np.array([10, 15, 15, 10, 15]),
    )

    # includes plane on the edge
    assert plane_intersects_aabb(
        plane_point=np.array([10, 10, 10, 10, 10]),
        plane_normal=np.array([10, 1, 0, 10, 0]),
        bounding_box_min=np.array([10, 10, 10, 10, 10]),
        bounding_box_max=np.array([10, 15, 15, 10, 15]),
    )

    # plane outside of the box
    assert not plane_intersects_aabb(
        plane_point=np.array([10, 10, 10, 10, 10]),
        plane_normal=np.array([10, 1, 0, 10, 0]),
        bounding_box_min=np.array([10, 11, 11, 10, 11]),
        bounding_box_max=np.array([10, 15, 15, 10, 15]),
    )


def test_generate_2D_grid():
    """Test a 2D grid with anisotropic spacing.

    Note the coordinates are ordered zyx.
    """
    shape = (3, 4)
    spacing = (2, 3)
    grid = generate_2d_grid(grid_shape=shape, grid_spacing=spacing)
    assert grid.shape == (*shape, 3)
    # x and y
    for spacing_index, axis in enumerate([1, 2]):
        vals = np.swapaxes(grid[..., axis], axis - 1, 0)
        # vals = grid[..., axis]
        np.testing.assert_allclose(vals[1:] - vals[:-1], spacing[spacing_index])
    # z axis
    z = grid[..., 0]
    assert np.all(z == 0)


def test_generate_3D_grid():
    shape = (2, 5, 3)
    spacing = (2, 1, 3)
    grid = generate_3d_grid(grid_shape=shape, grid_spacing=spacing)
    assert grid.shape == (*shape, 3)
    for axis in range(3):
        vals = np.swapaxes(grid[..., axis], axis, 0)
        np.testing.assert_allclose(vals[1:] - vals[:-1], spacing[axis])
