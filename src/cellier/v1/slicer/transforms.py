"""Classes and functions to express transformations."""

from abc import ABC, abstractmethod

import numpy as np
from psygnal import EventedModel
from pydantic import ConfigDict, field_validator
from pydantic_core.core_schema import ValidationInfo


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


class BaseTransform(EventedModel, ABC):
    """Base class for transformations."""

    @abstractmethod
    def map_coordinates(self, array):
        """Apply the transformation to coordinates.

        Parameters
        ----------
        array : np.ndarray
            (n, 4) Array to be transformed.
        """
        raise NotImplementedError

    @abstractmethod
    def imap_coordinates(self, array):
        """Apply the inverse transformation to coordinates.

        Parameters
        ----------
        array : np.ndarray
            (n, 4) array to be transformed.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError


class AffineTransform(BaseTransform):
    """Affine transformation.

    Parameters
    ----------
    matrix : np.ndarray
        The (4, 4) array encoding the affine transformation.

    Attributes
    ----------
    matrix : np.ndarray
        The (4, 4) array encoding the affine transformation.
    """

    matrix: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    @field_validator("matrix", mode="before")
    @classmethod
    def coerce_to_ndarray_float32(cls, v: str, info: ValidationInfo):
        """Coerce to a float32 numpy array."""
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.float32)
        return v.astype(np.float32)
