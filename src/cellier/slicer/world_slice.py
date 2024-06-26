"""classes to slice nD world data to displayed world data."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
from uuid import uuid4

import numpy as np

from cellier.models.scene import CoordinateSystem, DimsManager


class BaseWorldSlice(ABC):
    """Base class for World Slice data.

    These store the data required to specify the extents of the
    rendered view from the nD world space.
    """

    @property
    @abstractmethod
    def world_ndim(self) -> int:
        """The number of dimensions of the world."""
        raise NotImplementedError

    @property
    @abstractmethod
    def slice_ndim(self) -> int:
        """The number of dimensions in the sliced data."""
        raise NotImplementedError


@dataclass(frozen=True)
class AxisAligned2DWorldSlice(BaseWorldSlice):
    """Axis-aligned world 2D Slice data.

    These store the data required to specify the extents of the
    rendered view from the nD world space.

    Parameters
    ----------
    displayed_dimensions : Tuple[int, ...]
        The indices of the displayed dimensions.
        The order is matched to the displayed order.
    point : tuple of floats
        Dims position in world coordinates for each dimension.
    margin_negative : tuple of floats
        Negative margin in world units of the slice for each dimension.
    margin_positive : tuple of floats
        Positive margin in world units of the slice for each dimension.
    world_coordinate_system : CoordinateSystem
        The data model of the world coordinate system.
    id : str
        The unique identifier for the data store.
        The default value is a UUID4 generated hex string.
        In general, one should not specify and just let the ID
        be autogenerated.

    Attributes
    ----------
    displayed_dimensions : Tuple[int, ...]
        The indices of the displayed dimensions.
        The order is matched to the displayed order.
    point : tuple of floats
        Dims position in world coordinates for each dimension.
    margin_negative : tuple of floats
        Negative margin in world units of the slice for each dimension.
    margin_positive : tuple of floats
        Positive margin in world units of the slice for each dimension.
    world_coordinate_system : CoordinateSystem
        The data model of the world coordinate system.
    id : str
        The unique identifier for the data store.
    """

    displayed_dimensions: Tuple[int, ...]
    point: Tuple[int, ...]
    margin_negative: Tuple[float, ...]
    margin_positive: Tuple[float, ...]
    world_coordinate_system: CoordinateSystem

    # store a UUID to identify this specific scene.
    id: str = uuid4().hex

    @property
    def world_ndim(self) -> int:
        """The number of dimensions of the world."""
        return self.world_coordinate_system.ndim

    @property
    def slice_ndim(self) -> int:
        """The number of dimensions in the sliced data."""
        return 2

    @classmethod
    def from_dims(cls, dims: DimsManager):
        """Construct a world slice from the DimsManager object."""
        return cls(
            displayed_dimensions=dims.displayed_dimensions,
            point=dims.point,
            margin_negative=dims.margin_negative,
            margin_positive=dims.margin_positive,
            world_coordinate_system=dims.coordinate_system,
        )


@dataclass(frozen=True)
class AxisAligned3DWorldSlice(BaseWorldSlice):
    """Axis-aligned world 3D Slice data.

    These store the data required to specify the extents of the
    rendered view from the nD world space.

    Parameters
    ----------
    displayed_dimensions : Tuple[int, ...]
        The indices of the displayed dimensions.
        The order is matched to the displayed order.
    point : tuple of floats
        Dims position in world coordinates for each dimension.
    margin_negative : tuple of floats
        Negative margin in world units of the slice for each dimension.
    margin_positive : tuple of floats
        Positive margin in world units of the slice for each dimension.
    world_coordinate_system : CoordinateSystem
        The data model of the world coordinate system.
    id : str
        The unique identifier for the data store.
        The default value is a UUID4 generated hex string.
        In general, one should not specify and just let the ID
        be autogenerated.

    Attributes
    ----------
    displayed_dimensions : Tuple[int, ...]
        The indices of the displayed dimensions.
        The order is matched to the displayed order.
    point : tuple of floats
        Dims position in world coordinates for each dimension.
    margin_negative : tuple of floats
        Negative margin in world units of the slice for each dimension.
    margin_positive : tuple of floats
        Positive margin in world units of the slice for each dimension.
    world_coordinate_system : CoordinateSystem
        The data model of the world coordinate system.
    id : str
        The unique identifier for the data store.
    """

    displayed_dimensions: Tuple[int, ...]
    point: Tuple[int, ...]
    margin_negative: Tuple[float, ...]
    margin_positive: Tuple[float, ...]
    world_coordinate_system: CoordinateSystem

    # store a UUID to identify this specific scene.
    id: str = uuid4().hex

    @property
    def world_ndim(self) -> int:
        """The number of dimensions of the world."""
        return self.world_coordinate_system.ndim

    @property
    def slice_ndim(self) -> int:
        """The number of dimensions in the sliced data."""
        return 3

    @classmethod
    def from_dims(cls, dims: DimsManager):
        """Construct a world slice from the DimsManager object."""
        return cls(
            displayed_dimensions=dims.displayed_dimensions,
            point=dims.point,
            margin_negative=dims.margin_negative,
            margin_positive=dims.margin_positive,
            world_coordinate_system=dims.coordinate_system,
        )


@dataclass(frozen=True)
class ObliqueWorldSlice(BaseWorldSlice):
    """Data for an oblique world slice.

    Parameters
    ----------
    matrix : np.ndarray
        The affine matrix defining the slice.
    projection : str
        The function to use to project non-displayed dimensions.
    """

    matrix: np.ndarray
    projected_dimensions: Tuple[int, ...]
    projection: str = "sum"

    def world_ndim(self) -> int:
        """The number of dimensions in the world."""
        return self.matrix.shape[0] - 1

    def slice_ndim(self) -> int:
        """The number of dimensions in the slice."""
        return 2
