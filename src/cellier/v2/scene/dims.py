"""Dimension and coordinate system models for cellier v2."""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal, Union
from uuid import uuid4

from psygnal import EventedModel
from pydantic import UUID4, AfterValidator, Field, model_validator

from cellier.v2._state import (
    AxisAlignedSelectionState,
    DimsState,
    PlaneSelectionState,
)


class CoordinateSystem(EventedModel):
    """Names the scene's world coordinate axes.

    Parameters
    ----------
    name : str
        Human-readable name, e.g. ``"world"``.
    axis_labels : tuple[str, ...]
        One label per axis, e.g. ``("z", "y", "x")``.
    """

    name: str
    axis_labels: tuple[str, ...]


class AxisAlignedSelection(EventedModel):
    """Mutable selection model for axis-aligned slicing.

    Parameters
    ----------
    selector_type : Literal["axis_aligned"]
        Discriminator field.
    displayed_axes : tuple[int, ...]
        Indices into the coordinate system that are rendered.
        Length 2 → 2D; length 3 → 3D.
    slice_indices : dict[int, int]
        Mapping of axis index → slice value for non-displayed axes.
    """

    selector_type: Literal["axis_aligned"] = "axis_aligned"
    displayed_axes: tuple[int, ...]
    slice_indices: dict[int, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_displayed_rank(self) -> AxisAlignedSelection:
        """Display rank must be 2 or 3 (the GPU/camera constraint)."""
        n = len(self.displayed_axes)
        if n not in (2, 3):
            raise ValueError(
                f"displayed_axes must have length 2 or 3, got {n} "
                f"(displayed_axes={self.displayed_axes})"
            )
        if len(set(self.displayed_axes)) != n:
            raise ValueError(
                f"displayed_axes must contain distinct indices, "
                f"got {self.displayed_axes}"
            )
        return self

    def to_state(self) -> AxisAlignedSelectionState:
        """Return an immutable snapshot of this selection."""
        return AxisAlignedSelectionState(
            displayed_axes=self.displayed_axes,
            slice_indices=dict(self.slice_indices),
        )


class PlaneSelection(EventedModel):
    """Stub — not yet implemented.

    Parameters
    ----------
    selector_type : Literal["plane"]
        Discriminator field.
    """

    selector_type: Literal["plane"] = "plane"

    def to_state(self) -> PlaneSelectionState:
        """Return an immutable snapshot of this selection."""
        raise NotImplementedError("PlaneSelection is not yet implemented.")


SelectionType = Annotated[
    Union[AxisAlignedSelection, PlaneSelection],
    Field(discriminator="selector_type"),
]


class DimsManager(EventedModel):
    """Tracks which axes are displayed and the slice index for non-displayed axes.

    Single source of truth for render dimensionality.

    Parameters
    ----------
    id : UUID4
        Unique identifier. Auto-generated.
    coordinate_system : CoordinateSystem
        Defines the world axis names.
    selection : SelectionType
        The current axis selection (axis-aligned or plane).
    """

    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
    coordinate_system: CoordinateSystem
    selection: SelectionType

    @model_validator(mode="after")
    def _validate_axis_coverage(self) -> DimsManager:
        """Verify displayed + sliced axes cover all coordinate axes."""
        if isinstance(self.selection, AxisAlignedSelection):
            ndim = len(self.coordinate_system.axis_labels)
            covered = set(self.selection.displayed_axes) | set(
                self.selection.slice_indices.keys()
            )
            expected = set(range(ndim))
            if covered != expected:
                raise ValueError(
                    f"Axis coverage mismatch: displayed_axes | slice_indices.keys() "
                    f"= {covered}, expected {expected} for ndim={ndim}"
                )
        return self

    def model_post_init(self, __context: Any) -> None:
        """Wire event relays after model initialization."""
        self.coordinate_system.events.all.connect(self._on_coordinate_system_updated)
        self.selection.events.all.connect(self._on_selection_updated)

    def _on_coordinate_system_updated(self, info: Any) -> None:
        self.events.coordinate_system.emit(self.coordinate_system)

    def _on_selection_updated(self, info: Any) -> None:
        self.events.selection.emit(self.selection)

    def to_state(self) -> DimsState:
        """Return an immutable snapshot of the current dims state."""
        return DimsState(
            axis_labels=self.coordinate_system.axis_labels,
            selection=self.selection.to_state(),
        )
