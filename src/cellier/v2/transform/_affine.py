"""Affine coordinate transform for cellier v2."""

from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
from pydantic import field_serializer, field_validator
from typing_extensions import Self

from cellier.v2.transform._base import BaseTransform


def _to_homogeneous(coordinates: np.ndarray, ndim: int) -> np.ndarray:
    """Convert coordinates to homogeneous form for affine matrix multiplication.

    Parameters
    ----------
    coordinates : np.ndarray
        ``(n, ndim)`` or ``(n, ndim+1)`` array of points.
    ndim : int
        Expected number of spatial dimensions.

    Returns
    -------
    np.ndarray
        ``(n, ndim+1)`` array with a trailing 1 appended when needed.

    Raises
    ------
    ValueError
        If coordinates have an unexpected number of components.
    """
    coordinates = np.atleast_2d(coordinates)
    n_components = coordinates.shape[1]
    if n_components == ndim:
        return np.pad(coordinates, pad_width=((0, 0), (0, 1)), constant_values=1)
    elif n_components == ndim + 1:
        return coordinates
    else:
        raise ValueError(
            f"coordinates must have {ndim} or {ndim + 1} components per point, "
            f"got {n_components}"
        )


class AffineTransform(BaseTransform):
    """N-dimensional affine transformation using a homogeneous matrix.

    The matrix has shape ``(ndim+1, ndim+1)`` where ``ndim`` is the number
    of data dimensions.  Coordinates and scale/translation parameters
    follow data-axis order (e.g. ``(axis0, axis1, axis2)``).

    Instances are frozen (immutable). To change a transform, construct a
    new one and assign it to the visual's ``transform`` field.

    Composition uses left-to-right application order::

        (A @ B).map_coordinates(p) == B.map_coordinates(A.map_coordinates(p))

    Parameters
    ----------
    matrix : np.ndarray
        An ``(N, N)`` affine transformation matrix where ``N >= 2``.
    """

    matrix: np.ndarray

    @field_validator("matrix", mode="before")
    @classmethod
    def _coerce_to_ndarray_float32(cls, v: Any) -> np.ndarray:
        """Coerce to a float32 numpy array and validate shape."""
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
            raise ValueError(
                f"matrix must be square with shape (N, N) where N >= 2, got {arr.shape}"
            )
        return arr

    @field_serializer("matrix")
    def _serialize_matrix(self, v: np.ndarray) -> list:
        """Serialize the matrix to a nested list."""
        return v.tolist()

    # ── Cached properties ─────────────────────────────────────────────

    @cached_property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return self.matrix.shape[0] - 1

    @cached_property
    def inverse_matrix(self) -> np.ndarray:
        """Cached inverse of the affine matrix."""
        return np.linalg.inv(self.matrix).astype(np.float32)

    # ── Forward / inverse coordinate transforms ───────────────────────

    def map_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Apply the forward (data -> world) transform to coordinates.

        Coordinates are in data-axis order matching the numpy array shape
        convention (e.g. ``(axis0, axis1, axis2)``).

        Parameters
        ----------
        coordinates : np.ndarray
            ``(n, ndim)`` or ``(n, ndim+1)`` array of points.

        Returns
        -------
        np.ndarray
            ``(n, ndim)`` transformed coordinates.
        """
        return np.dot(_to_homogeneous(coordinates, self.ndim), self.matrix.T)[
            :, : self.ndim
        ]

    def imap_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Apply the inverse (world -> data) transform to coordinates.

        Coordinates are in data-axis order matching the numpy array shape
        convention (e.g. ``(axis0, axis1, axis2)``).

        Parameters
        ----------
        coordinates : np.ndarray
            ``(n, ndim)`` or ``(n, ndim+1)`` array of points.

        Returns
        -------
        np.ndarray
            ``(n, ndim)`` transformed coordinates.
        """
        return np.dot(_to_homogeneous(coordinates, self.ndim), self.inverse_matrix.T)[
            :, : self.ndim
        ]

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
            ``(n, ndim)`` normal vectors in data space.

        Returns
        -------
        np.ndarray
            ``(n, ndim)`` unit normal vectors in world space.
        """
        nd = self.ndim
        transformed = np.matmul(
            _to_homogeneous(normal_vector, nd), self.inverse_matrix
        )[:, :nd]
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
            ``(n, ndim)`` normal vectors in world space.

        Returns
        -------
        np.ndarray
            ``(n, ndim)`` unit normal vectors in data space.
        """
        nd = self.ndim
        transformed = np.matmul(_to_homogeneous(normal_vector, nd), self.matrix)[:, :nd]
        norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        return transformed / norms

    # ── Sub-transform extraction ──────────────────────────────────────

    def select_axes(self, axes: tuple[int, ...]) -> AffineTransform:
        """Extract a sub-transform for the given data axes.

        Returns a lower-dimensional transform containing only the
        rows/columns for the specified axes.  The output axis order
        matches the order of ``axes``.

        Parameters
        ----------
        axes : tuple[int, ...]
            Data axis indices to keep (e.g. ``(1, 2, 3)`` for z/y/x
            from a 4-D transform).

        Returns
        -------
        AffineTransform
            A ``len(axes)``-dimensional transform.
        """
        k = len(axes)
        m = np.eye(k + 1, dtype=np.float32)
        tc = self.ndim  # translation column index in source matrix
        for out_i, src_i in enumerate(axes):
            for out_j, src_j in enumerate(axes):
                m[out_i, out_j] = self.matrix[src_i, src_j]
            m[out_i, k] = self.matrix[src_i, tc]
        return AffineTransform(matrix=m)

    def swap_axes(self, permutation: tuple[int, ...]) -> AffineTransform:
        """Reorder the axes of this transform by an explicit permutation.

        Use this to convert a transform from one axis ordering to
        another (e.g. data axis order ``(z, y, x)`` to display axis
        order ``(x, y, z)``).  The permutation must be a permutation of
        ``range(self.ndim)``.

        Parameters
        ----------
        permutation : tuple[int, ...]
            ``permutation[i]`` is the source axis index whose row/column
            becomes output axis ``i``.

        Returns
        -------
        AffineTransform
            A transform of the same ``ndim`` with axes reordered.

        Raises
        ------
        ValueError
            If ``permutation`` is not a permutation of
            ``range(self.ndim)``.
        """
        n = self.ndim
        if len(permutation) != n or sorted(permutation) != list(range(n)):
            raise ValueError(
                f"permutation must be a permutation of range({n}), "
                f"got {permutation}"
            )
        return self.select_axes(permutation)

    def expand_dims(self, target_ndim: int) -> AffineTransform:
        """Embed this transform as the last ``self.ndim`` axes of a larger identity.

        Parameters
        ----------
        target_ndim : int
            Desired number of dimensions (must be >= ``self.ndim``).

        Returns
        -------
        AffineTransform
            A ``target_ndim``-dimensional transform with leading identity
            axes.

        Raises
        ------
        ValueError
            If ``target_ndim < self.ndim``.
        """
        if target_ndim < self.ndim:
            raise ValueError(
                f"target_ndim ({target_ndim}) must be >= self.ndim ({self.ndim})"
            )
        if target_ndim == self.ndim:
            return self
        m = np.eye(target_ndim + 1, dtype=np.float32)
        offset = target_ndim - self.ndim
        nd = self.ndim
        m[offset:target_ndim, offset:target_ndim] = self.matrix[:nd, :nd]
        m[offset:target_ndim, target_ndim] = self.matrix[:nd, nd]
        return AffineTransform(matrix=m)

    # ── Constructors ──────────────────────────────────────────────────

    @classmethod
    def identity(cls, ndim: int = 3) -> Self:
        """Return the identity transform.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions (default 3).
        """
        return cls(matrix=np.eye(ndim + 1, dtype=np.float32))

    @classmethod
    def from_scale(cls, scale: tuple[float, ...]) -> Self:
        """Return a scale-only transform with no translation.

        Parameters
        ----------
        scale : tuple[float, ...]
            Scale factors in data-axis order ``(axis0, axis1, ...)``.

        Returns
        -------
        AffineTransform
        """
        return cls.from_scale_and_translation(scale, tuple(0.0 for _ in scale))

    @classmethod
    def from_scale_and_translation(
        cls,
        scale: tuple[float, ...],
        translation: tuple[float, ...] | None = None,
    ) -> Self:
        """Create an AffineTransform from scale and translation parameters.

        Parameters
        ----------
        scale : tuple[float, ...]
            Scale factors in data-axis order ``(axis0, axis1, ...)``.
        translation : tuple[float, ...] or None
            Translation in data-axis order. Default is all zeros.

        Returns
        -------
        AffineTransform
        """
        ndim = len(scale)
        if translation is None:
            translation = tuple(0.0 for _ in range(ndim))
        if len(translation) != ndim:
            raise ValueError(
                f"scale and translation must have the same length, "
                f"got {len(scale)} and {len(translation)}"
            )
        matrix = np.eye(ndim + 1, dtype=np.float32)
        for i in range(ndim):
            matrix[i, i] = scale[i]
            matrix[i, ndim] = translation[i]
        return cls(matrix=matrix)

    @classmethod
    def from_translation(cls, translation: tuple[float, ...]) -> Self:
        """Create an AffineTransform from translation parameters.

        Parameters
        ----------
        translation : tuple[float, ...]
            Translation in data-axis order ``(axis0, axis1, ...)``.

        Returns
        -------
        AffineTransform
        """
        ndim = len(translation)
        matrix = np.eye(ndim + 1, dtype=np.float32)
        for i in range(ndim):
            matrix[i, ndim] = translation[i]
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

        Raises
        ------
        ValueError
            If the two transforms have different ``ndim``.
        """
        if self.ndim != other.ndim:
            raise ValueError(
                f"Cannot compose transforms with different ndim: "
                f"{self.ndim} vs {other.ndim}"
            )
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
