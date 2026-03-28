"""Affine coordinate transform for cellier v2."""

from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
from pydantic import field_serializer, field_validator
from typing_extensions import Self

from cellier.v2.transform._base import BaseTransform


def _to_vec4(coordinates: np.ndarray) -> np.ndarray:
    """Convert coordinates to homogeneous vec4 for affine matrix multiplication.

    Parameters
    ----------
    coordinates : np.ndarray
        ``(n, 3)`` or ``(n, 4)`` array of points.

    Returns
    -------
    np.ndarray
        ``(n, 4)`` array with a trailing 1 appended when input is 3-component.

    Raises
    ------
    ValueError
        If coordinates have neither 3 nor 4 components per point.
    """
    coordinates = np.atleast_2d(coordinates)
    n_components = coordinates.shape[1]
    if n_components == 3:
        return np.pad(coordinates, pad_width=((0, 0), (0, 1)), constant_values=1)
    elif n_components == 4:
        return coordinates
    else:
        raise ValueError(
            f"coordinates must have 3 or 4 components per point, " f"got {n_components}"
        )


class AffineTransform(BaseTransform):
    """Affine transformation using a 4x4 homogeneous matrix.

    Instances are frozen (immutable). To change a transform, construct a
    new one and assign it to the visual's ``transform`` field.

    Composition uses left-to-right application order::

        (A @ B).map_coordinates(p) == B.map_coordinates(A.map_coordinates(p))

    Parameters
    ----------
    matrix : np.ndarray
        A ``(4, 4)`` affine transformation matrix.
    """

    matrix: np.ndarray

    @field_validator("matrix", mode="before")
    @classmethod
    def _coerce_to_ndarray_float32(cls, v: Any) -> np.ndarray:
        """Coerce to a float32 numpy array and validate shape."""
        arr = np.asarray(v, dtype=np.float32)
        if arr.shape != (4, 4):
            raise ValueError(f"matrix must have shape (4, 4), got {arr.shape}")
        return arr

    @field_serializer("matrix")
    def _serialize_matrix(self, v: np.ndarray) -> list:
        """Serialize the matrix to a nested list."""
        return v.tolist()

    # ── Cached inverse ────────────────────────────────────────────────

    @cached_property
    def inverse_matrix(self) -> np.ndarray:
        """Cached inverse of the affine matrix."""
        return np.linalg.inv(self.matrix).astype(np.float32)

    # ── Forward / inverse coordinate transforms ───────────────────────

    def map_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Apply the forward transform to coordinates.

        Parameters
        ----------
        coordinates : np.ndarray
            ``(n, 3)`` or ``(n, 4)`` array of points.

        Returns
        -------
        np.ndarray
            ``(n, 3)`` transformed coordinates.
        """
        return np.dot(_to_vec4(coordinates), self.matrix.T)[:, :3]

    def imap_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Apply the inverse transform to coordinates.

        Parameters
        ----------
        coordinates : np.ndarray
            ``(n, 3)`` or ``(n, 4)`` array of points.

        Returns
        -------
        np.ndarray
            ``(n, 3)`` transformed coordinates.
        """
        return np.dot(_to_vec4(coordinates), self.inverse_matrix.T)[:, :3]

    # ── Normal vector transforms ──────────────────────────────────────

    def map_normal_vector(self, normal_vector: np.ndarray) -> np.ndarray:
        """Transform a normal vector from data space to world space.

        Notes
        -----
        Normals transform by the transpose-inverse of the point transform.
        Because the point transform is ``M``, the normal transform is
        ``(M^-1)^T``.  Multiplying row vectors on the left by ``M^-1``
        is equivalent to ``(M^-1)^T * n`` in column-vector convention.

        See Also
        --------
        imap_normal_vector : The inverse operation (world -> data).

        Parameters
        ----------
        normal_vector : np.ndarray
            ``(n, 3)`` normal vectors in data space.

        Returns
        -------
        np.ndarray
            ``(n, 3)`` unit normal vectors in world space.
        """
        transformed = np.matmul(_to_vec4(normal_vector), self.inverse_matrix)[:, :3]
        norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        return transformed / norms

    def imap_normal_vector(self, normal_vector: np.ndarray) -> np.ndarray:
        """Transform a normal vector from world space to data space.

        Notes
        -----
        This is the inverse of :meth:`map_normal_vector`.  The inverse
        normal transform uses ``M`` (the forward point matrix), because
        ``((M^-1)^-1)^T = M^T``, and left-multiplying by ``M`` is
        equivalent.

        See Also
        --------
        map_normal_vector : The forward operation (data -> world).

        Parameters
        ----------
        normal_vector : np.ndarray
            ``(n, 3)`` normal vectors in world space.

        Returns
        -------
        np.ndarray
            ``(n, 3)`` unit normal vectors in data space.
        """
        transformed = np.matmul(_to_vec4(normal_vector), self.matrix)[:, :3]
        norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        return transformed / norms

    # ── Constructors ──────────────────────────────────────────────────

    @classmethod
    def identity(cls) -> Self:
        """Return the identity transform."""
        return cls(matrix=np.eye(4, dtype=np.float32))

    @classmethod
    def from_scale(cls, scale: tuple[float, float, float]) -> Self:
        """Return a scale-only transform with no translation.

        Parameters
        ----------
        scale : tuple[float, float, float]
            Scale factors for ``(x, y, z)``.

        Returns
        -------
        AffineTransform
        """
        return cls.from_scale_and_translation(scale, (0.0, 0.0, 0.0))

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
            Scale factors for ``(x, y, z)`` dimensions.
        translation : tuple[float, float, float]
            Translation values for ``(x, y, z)`` dimensions.
            Default is ``(0, 0, 0)``.

        Returns
        -------
        AffineTransform
        """
        matrix = np.eye(4, dtype=np.float32)
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
            Translation values for ``(x, y, z)`` dimensions.

        Returns
        -------
        AffineTransform
        """
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 3] = translation[0]
        matrix[1, 3] = translation[1]
        matrix[2, 3] = translation[2]
        return cls(matrix=matrix)

    # ── Composition ───────────────────────────────────────────────────

    def compose(self, other: AffineTransform) -> AffineTransform:
        """Return a new transform: apply ``self`` first, then ``other``.

        ``self.compose(other).map_coordinates(p)`` is equivalent to
        ``other.map_coordinates(self.map_coordinates(p))``.

        Parameters
        ----------
        other : AffineTransform
            The transform to apply after ``self``.

        Returns
        -------
        AffineTransform
        """
        return AffineTransform(matrix=other.matrix @ self.matrix)

    def __matmul__(self, other: AffineTransform) -> AffineTransform:
        """Compose transforms: ``(A @ B)`` applies A first, then B.

        Parameters
        ----------
        other : AffineTransform
            The transform to apply after ``self``.

        Returns
        -------
        AffineTransform
        """
        return self.compose(other)
