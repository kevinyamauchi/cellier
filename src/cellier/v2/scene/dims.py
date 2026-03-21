"""Dimension and coordinate system models for cellier v2."""

from __future__ import annotations

import uuid
from typing import Annotated, Any
from uuid import uuid4

from psygnal import EventedModel
from pydantic import UUID4, AfterValidator, Field


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


class DimsManager(EventedModel):
    """Tracks which axes are displayed and the slice index for non-displayed axes.

    Single source of truth for render dimensionality.

    Parameters
    ----------
    id : UUID4
        Unique identifier. Auto-generated.
    coordinate_system : CoordinateSystem
        Defines the world axis names.
    displayed_axes : tuple[int, ...]
        Indices into ``coordinate_system.axis_labels`` currently rendered.
        Length 2 → 2D; length 3 → 3D.
    slice_indices : tuple[int, ...]
        One integer index per non-displayed axis, in the same axis order as
        ``coordinate_system``. Length must equal
        ``len(axis_labels) - len(displayed_axes)``.
    """

    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
    coordinate_system: CoordinateSystem
    displayed_axes: tuple[int, ...]
    slice_indices: tuple[int, ...]

    def model_post_init(self, __context: Any) -> None:
        """Wire coordinate system event relay after model initialization."""
        self.coordinate_system.events.all.connect(self._on_coordinate_system_updated)

    def _on_coordinate_system_updated(self, info: Any) -> None:
        self.events.coordinate_system.emit(self.coordinate_system)
