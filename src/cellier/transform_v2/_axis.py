"""Axes of a coordinate system (RFC-5 flavoured)."""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import UUID4, AliasChoices, BaseModel, ConfigDict, Field

AxisType = Literal["array", "space", "time", "channel", "coordinate", "displacement"]
"""The closed set of axis types defined by OME-NGFF RFC-5."""

AxisRef = str | UUID4
"""A reference to an axis, either by name or by id.

A ``str`` resolves through :meth:`CoordinateSystem.axis_by_name`, which
raises when the name is ambiguous within its system.  A ``UUID4``
resolves through :meth:`CoordinateSystem.index_of` and is never
ambiguous.
"""


class Axis(BaseModel):
    """A single axis of a coordinate system.

    Parameters
    ----------
    name : str
        Human-readable name, e.g. ``"z"``.  Names are not required to be
        unique within a coordinate system, but querying an ambiguous name
        raises.
    axis_type : AxisType
        The RFC-5 axis type.  Required: there is no default, because a
        channel or time axis silently typed as ``"space"`` fails far from
        where it was constructed.  Serialized as ``type``; both spellings
        parse.
    unit : str or None
        Free-form unit string.  RFC-5 recommends UDUNITS-2, but nothing
        here validates or interprets it.
    id : UUID4
        Unique identifier.  Auto-generated.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    axis_type: AxisType = Field(
        serialization_alias="type",
        validation_alias=AliasChoices("axis_type", "type"),
    )
    unit: str | None = None
    id: UUID4 = Field(default_factory=uuid4, frozen=True)
