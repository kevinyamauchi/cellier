"""Models for expressing and controlling the scene coordinate system."""

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Tuple, Union
from uuid import uuid4

import numpy as np
from psygnal import EmissionInfo, EventedModel
from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationInfo,
    field_serializer,
    field_validator,
)
from typing_extensions import Annotated

from cellier.types import (
    AxisAlignedSelectedRegion,
    CoordinateSpace,
    PlaneSelectedRegion,
    SelectedRegion,
)


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


@dataclass(frozen=True)
class FrozenCoordinateSystem:
    """Frozen coordinate system.

    Parameters
    ----------
    name : str
        The name of the coordinate system.
    axis_labels : tuple of str
        Tuple of labels for each dimension.
    """

    name: str
    axis_labels: Tuple[str, ...] = ()


@dataclass(frozen=True)
class _DimsState:
    """The current dims state (frozen).

    Parameters
    ----------
    id: str
        The ID of the dims manager this was generated from.
    coordinate_system: FrozenCoordinateSystem
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
    """

    id: str
    coordinate_system: FrozenCoordinateSystem
    displayed_dimensions: Tuple[int, ...]

    point: Tuple[float, ...] = ()
    range: Tuple[RangeTuple, ...] = ()
    margin_negative: Tuple[float, ...] = ()
    margin_positive: Tuple[float, ...] = ()


def _coerce_slice(raw_slice: tuple[int, int, int] | slice):
    if isinstance(raw_slice, tuple) or isinstance(raw_slice, list):
        return slice(*raw_slice)
    elif isinstance(raw_slice, slice):
        return raw_slice
    else:
        raise TypeError("Unrecognized type")


def _serialize_slice(to_serialize: slice) -> tuple[int | None, int | None, int | None]:
    return to_serialize.start, to_serialize.stop, to_serialize.step


# annotated slice for usage with the EventedModel
AnnotatedSlice = Annotated[
    slice,
    BeforeValidator(_coerce_slice),
    PlainSerializer(_serialize_slice, return_type=tuple),
]


class AxisAlignedRegionSelector(EventedModel):
    """A model for selecting an axis-aligned bounding box to display.

    Parameters
    ----------
    space_type : CoordinateSpace
        The coordinate space of the data.
        This is either in the data coordinates or the world coordinates.
    ordered_dims : tuple[int, ...]
        The dimension indices in their displayed order.
    n_displayed_dims : int
        The number of displayed dimensions.
    index_selection : tuple[Union[int, slice], ...]
        The selections for each dimension.
        Has the same order as ordered dims.
    """

    space_type: CoordinateSpace
    ordered_dims: tuple[int, ...]
    n_displayed_dims: Literal[2, 3]
    index_selection: tuple[Union[int, AnnotatedSlice], ...]

    selector_type: Literal["axis_aligned_region_selector"] = (
        "axis_aligned_region_selector"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_state(self) -> AxisAlignedSelectedRegion:
        """Get the current state as an immutable object.

        Returns
        -------
        AxisAlignedSelectedRegion
            The current state of the region selector.
            This is a frozen dataclass, so it is immutable.
        """
        return AxisAlignedSelectedRegion(
            space_type=self.space_type,
            ordered_dims=self.ordered_dims,
            n_displayed_dims=self.n_displayed_dims,
            index_selection=self.index_selection,
        )


class PlaneRegionSelector(EventedModel):
    """A model for selecting a 3D plane to display.

    Parameters
    ----------
    space_type : CoordinateSpace
        The coordinate space of the sample data.
        This is either in the data coordinates or the world coordinates.
    ordered_dims : tuple[int, ...]
        The dimension indices in their displayed order.
    n_displayed_dims : int
        The number of displayed dimensions.
    index_selection : tuple[Union[int, slice], ...]
        The selections for each dimension.
        Has the same order as ordered dims.
    point : np.ndarray
        The coordinate of the origin of the plane.
    plane_transform: np.ndarray
        The original plane has a normal vector [1, 0, 0]
        and an up vector [0, 1, 0].
        They should be ordered normal, up, right.
    extents : tuple[None | int, None | int]
        The extents of the plane along the up and right axes.
    """

    space_type: CoordinateSpace
    ordered_dims: tuple[int, ...]
    n_displayed_dims: Literal[2, 3]
    index_selection: tuple[Union[int, slice], ...]
    point: np.ndarray
    plane_transform: np.ndarray
    extents: tuple[None | int, None | int]

    model_config = ConfigDict(arbitrary_types_allowed=True)
    selector_type: Literal["plane_region_selector"]

    @field_validator("point", "plane_transform", mode="before")
    @classmethod
    def coerce_to_ndarray_float32(cls, v: str, info: ValidationInfo):
        """Coerce to a float32 numpy array."""
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.float32)
        return v.astype(np.float32)

    @field_serializer("point", "plane_transform")
    def serialize_ndarray(self, array: np.ndarray, _info) -> list:
        """Coerce numpy arrays into lists for serialization."""
        return array.tolist()

    def to_state(self) -> PlaneSelectedRegion:
        """Get the current state as an immutable object.

        Returns
        -------
        state : PlaneSelectedRegion
            The current state of the region selector.
            This is a frozen dataclass, so it is immutable.
        """
        return PlaneSelectedRegion(
            space_type=self.space_type,
            ordered_dims=self.ordered_dims,
            n_displayed_dims=self.n_displayed_dims,
            index_selection=self.index_selection,
            point=self.point,
            plane_transform=self.plane_transform,
            extents=self.extents,
        )


RegionSelectorType = Annotated[
    Union[AxisAlignedRegionSelector, PlaneRegionSelector],
    Field(discriminator="selector_type"),
]


@dataclass(frozen=True)
class DimsState:
    """The current dims state (frozen).

    Parameters
    ----------
    id: str
        The ID of the dims manager this was generated from.
    coordinate_system: FrozenCoordinateSystem
        The coordinate system the dimensions belong to.
    range : tuple[RangeTuple, ...]
        List of tuples (min, max, step), one for each dimension in world
        coordinates space. Lower and upper bounds are inclusive.
    selection : SelectedRegion
        The current region selected for rendering.
    """

    id: str
    coordinate_system: FrozenCoordinateSystem
    range: tuple[RangeTuple, ...]
    selection: SelectedRegion


class DimsManager(EventedModel):
    """Model of the dimensions of a scene.

    Parameters
    ----------
    id : str
        The unique ID of this DimsManager instance.
    coordinate_system: CoordinateSystem
        The coordinate system the dimensions belong to.
    range : tuple[RangeTuple, ...]
        List of tuples (min, max, step), one for each dimension in world
        coordinates space. Lower and upper bounds are inclusive.
    selection : SelectedRegion
        The selection to display for each dimension.
    """

    coordinate_system: CoordinateSystem
    range: Tuple[RangeTuple, ...]
    selection: RegionSelectorType

    id: str = Field(default_factory=lambda: uuid4().hex)

    def model_post_init(self, __context: Any) -> None:
        """Connect the selection model events.

        This is a pydantic model method that is automatically called
        after the model is initialized.
        """
        self.selection.events.all.connect(self._on_selection_changed)

    @property
    def ndisplay(self) -> int:
        """The number of displayed dimensions."""
        return self.selection.n_displayed_dims

    def to_state(self) -> DimsState:
        """Return the current state of the dims as a DimsState object.

        Returns
        -------
        state : DimsState
            The current state of the dims.
            This is a frozen dataclass, so it is immutable.
        """
        return DimsState(
            id=self.id,
            coordinate_system=FrozenCoordinateSystem(
                name=self.coordinate_system.name,
                axis_labels=self.coordinate_system.axis_labels,
            ),
            range=self.range,
            selection=self.selection.to_state(),
        )

    def update_state(self, new_state: dict) -> None:
        """Update the state of the dims.

        This is often used as a callback for when
        the dims controls update.
        """
        # remove the id field from the new state if present
        new_state.pop("id", None)
        # update the visual with the new state
        self.update(new_state)

    def _on_selection_changed(self, event: EmissionInfo):
        """Callback to relay the signal emitted on the RegionSelector.

        This is connected in the model_post_init method.
        """
        self.events.selection.emit()
