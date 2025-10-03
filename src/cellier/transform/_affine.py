"""Classes and functions to express transformations."""

import numpy as np
from pydantic import ConfigDict, field_serializer, field_validator
from pydantic_core.core_schema import ValidationInfo
from typing_extensions import Self

from cellier.transform._base import BaseTransform


def to_vec4(coordinates: np.ndarray) -> np.ndarray:
    """Convert coordinates to vec4 to make compatible with an affine matrix."""
    coordinates = np.atleast_2d(coordinates)

    ndim = coordinates.shape[1]
    if ndim == 3:
        # add a 1 in the fourth dimension.
        return np.pad(coordinates, pad_width=((0, 0), (0, 1)), constant_values=1)

    elif ndim == 4:
        return coordinates

    else:
        raise ValueError(f"Coordinates should be 3D or 4D, coordinates were {ndim}D")


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
        return np.dot(to_vec4(coordinates), self.matrix.T)[:, :3]

    def imap_coordinates(self, coordinates: np.ndarray):
        """Apply the inverse transformation to coordinates."""
        return np.dot(to_vec4(coordinates), np.linalg.inv(self.matrix).T)[:, :3]

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
        normal_transform = np.linalg.inv(self.matrix)
        transformed_vector = np.matmul(to_vec4(normal_vector), normal_transform)[:, :3]

        return transformed_vector / np.expand_dims(
            np.linalg.norm(transformed_vector, axis=1), axis=1
        )

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
        normal_transform = self.matrix
        transformed_vector = np.matmul(to_vec4(normal_vector), normal_transform)[:, :3]

        return transformed_vector / np.expand_dims(
            np.linalg.norm(transformed_vector, axis=1), axis=1
        )

    @field_validator("matrix", mode="before")
    @classmethod
    def coerce_to_ndarray_float32(cls, v: str, info: ValidationInfo):
        """Coerce to a float32 numpy array."""
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.float32)
        return v.astype(np.float32)

    @field_serializer("matrix")
    @classmethod
    def serialize_matrix(cls, v: np.ndarray) -> list:
        """Serialize the matrix to a list."""
        return v.tolist()

    @classmethod
    def from_scale_and_translation(
        cls,
        scale: tuple[float, float, float],
        translation: tuple[float, float, float] = (0, 0, 0),
    ) -> Self:
        """Create an AffineTransform from scale and translation parameters.

        Parameters
        ----------
        scale : tuple[float, float, float]
            Scale factors for (x, y, z) dimensions.
        translation : tuple[float, float, float]
            Translation values for (x, y, z) dimensions. Default is (0, 0, 0).

        Returns
        -------
        AffineTransform
            The affine transformation with the specified scale and translation.
        """
        matrix = np.eye(4)
        matrix[0, 0] = scale[0]
        matrix[1, 1] = scale[1]
        matrix[2, 2] = scale[2]
        matrix[0, 3] = translation[0]
        matrix[1, 3] = translation[1]
        matrix[2, 3] = translation[2]

        return cls(matrix=matrix)

    @classmethod
    def from_translation(cls, translation: tuple[float, float, float]) -> Self:
        """Create an AffineTransform from translation parameters.

        Parameters
        ----------
        translation : tuple[float, float, float]
            Translation values for (x, y, z) dimensions.

        Returns
        -------
        AffineTransform
            The affine transformation with the specified translation.
        """
        matrix = np.eye(4)
        matrix[0, 3] = translation[0]
        matrix[1, 3] = translation[1]
        matrix[2, 3] = translation[2]
        return cls(matrix=matrix)
