from abc import ABC, abstractmethod

import numpy as np
from psygnal import EventedModel


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
