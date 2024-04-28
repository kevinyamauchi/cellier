"""Classes and functions to express transformations."""

from abc import ABC, abstractmethod

import numpy as np
from psygnal import EventedModel
from pydantic import ConfigDict, field_validator
from pydantic_core.core_schema import ValidationInfo


class BaseTransform(EventedModel, ABC):
    """Base class for transformations."""

    @abstractmethod
    def map(self, array):
        """Apply the transformation to coordinates.

        Parameters
        ----------
        array : np.ndarray
            (n, 4) Array to be transformed.
        """
        raise NotImplementedError

    @abstractmethod
    def imap(self, array):
        """Apply the inverse transformation to coordinates.

        Parameters
        ----------
        array : np.ndarray
            (n, 4) array to be transformed.
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

    def map(self, array: np.ndarray):
        """Apply the transformation to coordinates.

        Parameters
        ----------
        array : np.ndarray
            (n, 4) array to be transformed.
        """
        return np.dot(array, self.matrix)

    def imap(self, array: np.ndarray):
        """Apply the inverse transformation to coordinates.

        Parameters
        ----------
        array : np.ndarray
            (n, 4) array to be transformed.
        """
        return np.dot(array, np.linalg.inv(self.matrix))

    @field_validator("matrix", mode="before")
    @classmethod
    def coerce_to_ndarray_float32(cls, v: str, info: ValidationInfo):
        """Coerce to a float32 numpy array."""
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.float32)
        return v.astype(np.float32)
