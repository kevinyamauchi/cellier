"""Example showing how oblique slicing works on points."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def make_plane_mesh(
    point: np.ndarray, normal: np.ndarray, width: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Make a mesh to visualize a plane."""
    pass


def to_vec4(coordinates: np.ndarray) -> np.ndarray:
    """Convert coordinates to vec4 to make compatible with an affine matrix."""
    coordinates = np.atleast_2d(coordinates)

    ndim = coordinates.shape[1]
    if ndim == 3:
        # add a 1 in the fourth dimension.
        return np.pad(coordinates, pad_width=((0, 0), (0, 1)), constant_values=1)

    elif coordinates.shape == 4:
        return coordinates

    else:
        raise ValueError(f"Coordinates should be 3D or 4D, coordinates were {ndim}D")


def compute_plane_coefficients(
    point: np.ndarray, normal_vector: np.ndarray
) -> Tuple[float, float, float, float]:
    """Compute the coefficients for the 3D plane equation..

    Ax_0 + Bx_1 + Cx_2 + d = 0


    Parameters
    ----------
    point : np.ndarray
        A point on the plane.
    normal_vector : np.ndarray
        The unit normal vector of the plane.

    Returns
    -------
    a : float
        The coefficient for the 0th axis.
    b : float
        The coefficient for the 1st axis.
    c : float
        The coefficient for the 2nd axis.
    d : float
        The final coefficient.
    """
    a, b, c = normal_vector
    d = np.dot(-point, normal_vector)
    return a, b, c, d


def points_in_front_of_plane(
    point_coordinates: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray
) -> np.ndarray:
    """Determine if points are in front of a plane.

    In front is defined as the side the normal vector is point to.

    Parameters
    ----------
    point_coordinates : np.ndarray
        The coordinates of the points to test.
    plane_point : np.ndarray
        A point on the plane.
    plane_normal : np.ndarray
        The unit normal vector of the plane. Points on the side
        this normal vector is pointing to are considered "in front".

    Returns
    -------
    in_front : np.ndarray
        (n,) boolean array where each elemennt is True if the index-matched
        point is "in front" of the plane. "in front" is defined as on the same
        side the normal vector is pointing.
    """
    plane_coefficients = compute_plane_coefficients(
        point=plane_point, normal_vector=plane_normal
    )
    return (
        plane_coefficients[0] * point_coordinates[:, 0]
        + plane_coefficients[1] * point_coordinates[:, 1]
        + plane_coefficients[2] * point_coordinates[:, 2]
        + plane_coefficients[3]
    ) >= 0


@dataclass(frozen=True)
class Oblique2DWorldSliceRequest:
    """Data to request an oblique 2D slice in world coordinates.

    For convenience, I am using numpy arrays - may use tuples in
    the future for serializabiltiy.

    Parameters
    ----------
    point: np.ndarray
        The point in nD world coordinates the plane is centered on.
        This will be the (0, 0) coordinate of the resulting 2D slice.
    normal_vector: np.ndarray
        The normal unit vector of the plane.
        This is in the 3D spatial coordinate system described by spatial_dims.
    up_vector: np.ndarray
        The up unit vector of the plane. This will be the (0, 1) direction
        of the resulting 2D slice.
        This is in the 3D spatial coordinate system described by spatial_dims.
    spatial_dims : Tuple[int, int, int]
        The axis indices of the nD world coordinate system for the
        3D spatial axes comprising the spatial coordinate system.
    slice_positive_extent : float
        The extent of the slice in the positive direction of the normal vector.
    slice_negative_extent  : float
        The extent of the slice in the negative direction of the normal vector.
    """

    point: np.ndarray
    normal_vector: np.ndarray
    up_vector: np.ndarray
    spatial_dims: Tuple[int, int, int]
    slice_positive_extent: float
    slice_negative_extent: float


@dataclass(frozen=True)
class Oblique2DDataSliceRequest:
    """Data to request an oblique 2D slice in world coordinates.

    For convenience, I am using numpy arrays - may use tuples in
    the future for serializabiltiy.

    Parameters
    ----------
    point: np.ndarray
        The point in nD data coordinates the plane is centered on.
        This will be the (0, 0) coordinate of the resulting 2D slice.
    normal_vector: np.ndarray
        The normal unit vector of the plane.
        This is in the 3D spatial coordinate system described by spatial_dims.
    up_vector: np.ndarray
        The up unit vector of the plane. This will be the (0, 1) direction
        of the resulting 2D slice.
        This is in the 3D spatial coordinate system described by spatial_dims.
    spatial_dims : Tuple[int, int, int]
        The axis indices of nD data coordinate system for
        the 3D spatial axes comprising the spatial coordinate system.
    slice_positive_extent : float
        The extent of the slice in the positive direction of the normal vector.
    slice_negative_extent  : float
        The extent of the slice in the negative direction of the normal vector.
    """

    point: np.ndarray
    normal_vector: np.ndarray
    up_vector: np.ndarray
    spatial_dims: Tuple[int, int, int]
    slice_positive_extent: float
    slice_negative_extent: float


@dataclass
class AffineTransform:
    """Affine transformation class."""

    matrix: np.ndarray

    def map_coordinates(self, coordinates: np.ndarray):
        """Apply the transformation to coordinates."""
        return np.dot(to_vec4(coordinates), self.matrix)[:, :3]

    def imap_coordinates(self, coordinates: np.ndarray):
        """Apply the inverse transformation to coordinates."""
        return np.dot(to_vec4(coordinates), np.linalg.inv(self.matrix))[:, :3]

    def map_normal_vector(self, normal_vector: np.ndarray):
        """Apply the transform to a normal vector defining an orientation.

        For example, this would be used to a plane normal.

        https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-normals.html

        Parameters
        ----------
        normal_vector : np.ndarray
            The normal vector(s) to be transformed.

        Returns
        -------
        transformed_vector : np.ndarray
            The transformed normal vectors as a unit vector.
        """
        normal_transform = np.linalg.inv(self.matrix).T
        transformed_vector = np.matmul(to_vec4(normal_vector), normal_transform)[:, :3]

        return transformed_vector / np.linalg.norm(transformed_vector, axis=1)

    def imap_normal_vector(self, normal_vector: np.ndarray):
        """Apply the inverse transform to a normal vector defining an orientation.

        For example, this would be used to a plane normal.

        https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-normals.html

        Parameters
        ----------
        normal_vector : np.ndarray
            The normal vector(s) to be transformed.

        Returns
        -------
        transformed_vector : np.ndarray
            The transformed normal vectors as a unit vector.
        """
        normal_transform = self.matrix.T
        transformed_vector = np.matmul(to_vec4(normal_vector), normal_transform)[:, :3]

        return transformed_vector / np.linalg.norm(transformed_vector, axis=1)


# make 4D points
n_points = 100
ndim_data = 4
rng = np.random.default_rng()
point_spatial_coordinates = 50 * rng.random((n_points, 3))
point_time_coordinates = 5 * rng.random((n_points, 1))
point_coordinates = np.column_stack((point_time_coordinates, point_spatial_coordinates))

# define the transformation from data to world coordinates
data_to_world_transform = AffineTransform(matrix=np.eye(4))

# world to spatial dims mapping
# we have a 5D world and a 4D point coordinates
world_to_data_axis_mapping = {0: None, 1: 0, 2: 1, 3: 2, 4: 3}

data_to_world_axis_mapping = {0: 1, 1: 2, 2: 3, 3: 4}

# define the world slice
world_slice = Oblique2DWorldSliceRequest(
    point=np.array([1, 0, 25, 25, 25]),
    normal_vector=np.array([1, 0, 0]),
    up_vector=np.array([0, 1, 0]),
    spatial_dims=(2, 3, 4),
    slice_positive_extent=0.5,
    slice_negative_extent=0.5,
)

# convert the world slice to data coordinates
# get the spatial dims in data coordinates
spatial_dims_data = [
    world_to_data_axis_mapping[axis_index] for axis_index in world_slice.spatial_dims
]

# get the point in data coordinates.
# first we get just the spatial components of the world point.
spatial_dims_world = list(world_slice.spatial_dims)
point_world_spatial = world_slice.point[spatial_dims_world]
point_data_spatial = np.squeeze(
    data_to_world_transform.imap_coordinates(point_world_spatial)
)
point_data = np.empty((ndim_data,))
for data_index in range(ndim_data):
    if data_index in spatial_dims_data:
        spatial_index = spatial_dims_data.index(data_index)
        point_data[data_index] = point_data_spatial[spatial_index]
    else:
        world_index = data_to_world_axis_mapping[data_index]
        point_data[data_index] = world_slice.point[world_index]


# get the vectors in data coordinates
normal_data = np.squeeze(
    data_to_world_transform.imap_normal_vector(world_slice.normal_vector)
)
up_data = np.squeeze(data_to_world_transform.imap_normal_vector(world_slice.up_vector))

data_slice = Oblique2DDataSliceRequest(
    point=point_data,
    normal_vector=normal_data,
    up_vector=up_data,
    spatial_dims=tuple(spatial_dims_data),
    slice_positive_extent=0.5,
    slice_negative_extent=0.5,
)
print(data_slice)

# get the data

# determine if points on the correct side of the positive plane
positive_plane_point = data_slice.point + (
    data_slice.slice_positive_extent * data_slice.normal_vector
)
in_front_positive_plane = points_in_front_of_plane(
    point_coordinates=point_coordinates,
    plane_point=positive_plane_point,
    plane_normal=-data_slice.normal_vector,
)

# determine if the point is ont he correct side of the negative points
negative_plane_point = data_slice.point - (
    data_slice.slice_positive_extent * data_slice.normal_vector
)
in_front_negative_plane = points_in_front_of_plane(
    point_coordinates=point_coordinates,
    plane_point=negative_plane_point,
    plane_normal=data_slice.normal_vector,
)

# points in the slice are on the correct side of both planes
points_in_slice_mask = np.logical_and(in_front_positive_plane, in_front_negative_plane)
