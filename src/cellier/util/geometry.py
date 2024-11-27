"""Functions for computing on geometries."""

import numpy as np

# indices to construct the lines of a frustum from an array
# of the corners of the frustum. Each element of this list are the indices
# for an edge of the frustum.
# the corner coordinates should be in an array shaped (2, 4, 3).
# The first axis corresponds to the frustums plane (near, far).
# The second to the corner within the plane
#   ((left, bottom), (right, bottom), (right, top), (left, top))
# The third to the world position of that corner.
frustum_edge_indices = [
    ((0, 0), (0, 1)),
    ((0, 1), (0, 2)),
    ((0, 2), (0, 3)),
    ((0, 3), (0, 0)),
    ((1, 0), (1, 1)),
    ((1, 1), (1, 2)),
    ((1, 2), (1, 3)),
    ((1, 3), (1, 0)),
    ((0, 0), (1, 0)),
    ((0, 1), (1, 1)),
    ((0, 2), (1, 2)),
    ((0, 3), (1, 3)),
]


frustum_plane_point_indices = [
    ((0, 2), (0, 1), (0, 0)),  # near
    ((1, 2), (1, 3), (1, 0)),  # far
    ((0, 3), (0, 0), (1, 0)),  # left
    ((1, 2), (1, 1), (0, 1)),  # right
    ((1, 2), (0, 2), (0, 3)),  # top
    ((1, 1), (1, 0), (0, 0)),  # bottom
]

# corner indices for the edges of the near and far plane of the frustum.
# edges are:
# - near-bottom
# - near-right
# - near-top
# - near-left
# - far-bottom
# - far-right
# - far-top
# - far-left
# corners must be ordered the same as above.
near_far_plane_edge_indices = [
    ((0, 0), (0, 1)),
    ((0, 1), (0, 2)),
    ((0, 2), (0, 3)),
    ((0, 3), (0, 0)),
    ((1, 0), (1, 1)),
    ((1, 1), (1, 2)),
    ((1, 2), (1, 3)),
    ((1, 3), (1, 0)),
]


def frustum_edges_from_corners(corners: np.ndarray) -> np.ndarray:
    """Get an array of the frustum edges from the corners.

    Parameters
    ----------
    corners : np.ndarray
        The coordinates of the corners of the frustum.
        The corner coordinates should be in an array shaped (2, 4, 3).
        The first axis corresponds to the frustum plane (near, far).
        The second to the corner within the plane ((left, bottom), (right, bottom),
        (right, top), (left, top))
        The third to the world position of that corner.

    Returns
    -------
    np.ndarray
        The frustum edges. The array has shape (12, 2, 3) where the
        first axis is the edge, the second axis is the (start, end) point,
        and the third axis contains the corner coordinates.
        The array is float32.
    """
    return np.array(
        [
            [
                corners[start_end_index[0], start_end_index[1], :]
                for start_end_index in line_data
            ]
            for line_data in frustum_edge_indices
        ],
        dtype=np.float32,
    )


def near_far_plane_edge_lengths(corners: np.ndarray) -> np.ndarray:
    """Calculate the length of each edges of the near and far plane of the frustum."""
    edge_lengths = np.zeros((2, 4))
    for near_far_index in range(2):
        for edge_index in range(4):
            edge_indices = frustum_edge_indices[(near_far_index * 4) + edge_index]
            edge_vector = (
                corners[edge_indices[0][0], edge_indices[0][1], :]
                - corners[edge_indices[1][0], edge_indices[1][1], :]
            )
            edge_lengths[near_far_index, edge_index] = np.linalg.norm(edge_vector)

    return edge_lengths


def compute_plane_parameters(point_0, point_1, point_2):
    """Compute the plane parameters from 3 points on the plane.

    The planes are parameterized by a * x_0 + b * x_1 + c * x_2 + d = 0
    The normal of the plane is pointing out of the side of the plane where
    the points are counter-clockwise when looking at the plane.

    Parameters
    ----------
    point_0 : np.ndarray
        The first point on the plane.
    point_1 : np.ndarray
        The second point on the plane.
    point_2 : np.ndarray
        The third point on the plane.

    Returns
    -------
    np.ndarray
        The parameters from the plane equation (a, b, c, d) where
        a * x_0 + b * x_1 + c * x_2 + d = 0
    """
    # Compute the unit normal vector using the cross product of two edges
    v1 = point_1 - point_0
    v2 = point_2 - point_0
    normal = np.cross(v1, v2)
    unit_normal = normal / np.linalg.norm(normal)

    # Compute d by substituting a point into the plane equation
    d = -np.dot(unit_normal, point_0)

    # Return the plane coefficients
    return np.append(unit_normal, d)


def frustum_planes_from_corners(corners: np.ndarray) -> np.ndarray:
    """Compute the plane coefficient for each plane of a frustum from its corners.

    The planes are parameterized by a * x_0 + b * x_1 + c * x_2 + d = 0

    Parameters
    ----------
    corners : np.ndarray
        The coordinates of the corners of the frustum.
        The corner coordinates should be in an array shaped (2, 4, 3).
        The first axis corresponds to the frustum plane (near, far).
        The second to the corner within the plane
            ((left, bottom), (right, bottom), (right, top), (left, top))
        The third to the world position of that corner.

    Returns
    -------
    np.ndarray
        The plane coefficient for each plane in the frustum.
        The first axis is the plane and the second axis are the coefficients
        (a, b, c, d).
        The planes are ordered near, far, left, right, top, bottom.
    """
    return np.array(
        [
            compute_plane_parameters(
                corners[point_indices[0][0], point_indices[0][1], :],
                corners[point_indices[1][0], point_indices[1][1], :],
                corners[point_indices[2][0], point_indices[2][1], :],
            )
            for point_indices in frustum_plane_point_indices
        ]
    )


def points_in_frustum(points: np.ndarray, planes: np.ndarray) -> np.ndarray:
    """Check which points are inside a frustum defined by planes.

    Note that points on a plane are considered inside the frustum.

    Parameters
    ----------
    points : np.ndarray
        (M, 3) array where each row is a 3D point.
    planes : np.ndarray
        (N, 4) array where each row contains the coefficients [a, b, c, d] for a plane.

    Returns
    -------
    inside_mask : np.ndarray
        (M,) boolean array where values are True if the point is inside of the frustum.
    """
    # Compute distances of points to all planes: (M, 3) · (N, 3).T + (N,)
    distances = np.dot(points, planes[:, :3].T) + planes[:, 3]

    # Check if the point is in front of all planes (distance >= 0 for all planes)
    inside_mask = np.all(distances >= 0, axis=1)

    return inside_mask
