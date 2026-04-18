"""Scene model for cellier v2."""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal
from uuid import uuid4

from psygnal import EventedModel
from pydantic import UUID4, AfterValidator, Field

from cellier.v2.scene.canvas import Canvas
from cellier.v2.scene.dims import DimsManager
from cellier.v2.visuals._types import VisualType


class Scene(EventedModel):
    """One rendered scene.

    Parameters
    ----------
    id : UUID4
        Unique identifier. Auto-generated.
    name : str
        Human-readable name, e.g. ``"main"``.
    dims : DimsManager
        Dimension manager; single source of truth for render dimensionality.
    visuals : list[VisualType]
        Discriminated union of visual model types.
    canvases : dict[UUID4, Canvas]
        Keyed by ``canvas.id``.
    render_modes : set[Literal["2d", "3d"]]
        Which rendering modes visuals added to this scene should support.
        Defaults to ``{"2d", "3d"}``.
    lighting : Literal["none", "default"]
        ``"none"`` (default) or ``"default"``.  Pass ``"default"`` to add
        ambient and directional lights — required for MeshPhongAppearance.
    """

    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
    name: str
    dims: DimsManager
    visuals: list[VisualType] = Field(default_factory=list)
    canvases: dict[
        UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))],
        Canvas,
    ] = Field(default_factory=dict)
    render_modes: set[Literal["2d", "3d"]] = Field(default_factory=lambda: {"2d", "3d"})
    lighting: Literal["none", "default"] = "none"

    def model_post_init(self, __context: Any) -> None:
        """Wire dims and visual event relays after model initialization."""
        self.dims.events.all.connect(self._on_dims_updated)
        for visual in self.visuals:
            visual.events.all.connect(
                lambda info, v=visual: self.events.visuals.emit(self.visuals)
            )

    def _on_dims_updated(self, info: Any) -> None:
        self.events.dims.emit(self.dims)
