"""Models for expressing and controlling the scene coordinate system."""

from typing import NamedTuple, Tuple

from psygnal import EventedModel
from pydantic import ValidationInfo, field_validator


class CoordinateSystem(EventedModel):
    """Model for a coordinate system.

    Parameters
    ----------
    name : str
        The name of the coordinate system.
    axis_labels : tuple of str
        Tuple of labels for each dimension.
    """

    name: str
    axis_labels: Tuple[str, ...] = ()

    @field_validator("axis_labels", mode="before")
    def coerce_to_tuple(cls, v, info: ValidationInfo):
        """Coerce the axis label names to a tuple."""
        if not isinstance(v, tuple):
            v = tuple(v)
        return v

    @property
    def ndim(self) -> int:
        """Number of dimensions in the coordinate system.

        Returns
        -------
        int
            The number of dimensions in the coordinate system.
        """
        return len(self.axis_labels)


class RangeTuple(NamedTuple):
    """Data item to express a range of discrete values."""

    start: float
    stop: float
    step: float


class DimsManager(EventedModel):
    """Model of the dimensions of a scene.

    The DimsManager is assigned to a single scene.

    Parameters
    ----------
    coordinate_system: CoordinateSystem
        The coordinate system the dimensions belong to.
    displayed_dimensions: Tuple[int,...]
        The names of the displayed dimensions. The order indicates the order of
        the axes. For example [0, 2, 1] means that 0 is the 0th dimension,
        2 is the 1st dimension, and 1 is the 2nd dimension.
    range : tuple of 3-tuple of float
        List of tuples (min, max, step), one for each dimension in world
        coordinates space. Lower and upper bounds are inclusive.
    point : tuple of floats
        Dims position in world coordinates for each dimension.
    margin_negative : tuple of floats
        Negative margin in world units of the slice for each dimension.
    margin_positive : tuple of floats
        Positive margin in world units of the slice for each dimension.
    order : tuple of int
        Tuple of ordering the dimensions, where the last dimensions are rendered.


    Attributes
    ----------
    ndisplay : int
        Number of displayed dimensions.
    """

    coordinate_system: CoordinateSystem
    displayed_dimensions: Tuple[int, ...]

    point: Tuple[float, ...] = ()
    range: Tuple[RangeTuple, ...] = ()
    margin_negative: Tuple[float, ...] = ()
    margin_positive: Tuple[float, ...] = ()

    @field_validator(
        "displayed_dimensions",
        "point",
        "margin_negative",
        "margin_positive",
        mode="before",
    )
    def coerce_to_tuple(cls, v, info: ValidationInfo):
        """Coerce the axis label names to a tuple."""
        if not isinstance(v, tuple):
            v = tuple(v)
        return v

    @property
    def ndisplay(self) -> int:
        """The number of displayed dimensions."""
        return len(self.displayed_dimensions)
