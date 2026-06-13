"""Abstract base class for v2 transforms."""

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, ConfigDict


class BaseTransform(BaseModel, ABC):
    """Base class for coordinate transforms.

    All v2 transforms are frozen pydantic models. The signal for a transform
    change travels on the visual's ``EventedModel`` field, not on the
    transform itself.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @abstractmethod
    def map_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Apply the forward transform to coordinates.

        Parameters
        ----------
        coordinates : np.ndarray
            ``(n, ndim)`` or ``(n, ndim+1)`` array of points.

        Returns
        -------
        np.ndarray
            ``(n, ndim)`` transformed coordinates.
        """
        raise NotImplementedError

    @abstractmethod
    def imap_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Apply the inverse transform to coordinates.

        Parameters
        ----------
        coordinates : np.ndarray
            ``(n, ndim)`` or ``(n, ndim+1)`` array of points.

        Returns
        -------
        np.ndarray
            ``(n, ndim)`` transformed coordinates.
        """
        raise NotImplementedError

    @abstractmethod
    def map_normal_vector(self, normal_vector: np.ndarray) -> np.ndarray:
        """Transform a normal vector from data space to world space.

        Parameters
        ----------
        normal_vector : np.ndarray
            The normal vector(s) to be transformed.

        Returns
        -------
        np.ndarray
            The transformed normal vectors as unit vectors.
        """
        raise NotImplementedError

    @abstractmethod
    def imap_normal_vector(self, normal_vector: np.ndarray) -> np.ndarray:
        """Transform a normal vector from world space to data space.

        Parameters
        ----------
        normal_vector : np.ndarray
            The normal vector(s) to be transformed.

        Returns
        -------
        np.ndarray
            The transformed normal vectors as unit vectors.
        """
        raise NotImplementedError
